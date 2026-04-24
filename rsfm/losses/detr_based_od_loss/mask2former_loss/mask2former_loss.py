import torch
from torch import nn
from rsfm.losses.detr_based_od_loss.detr_loss import is_dist_avail_and_initialized, get_world_size
from rsfm.utils import nested_tensor_from_tensor_list
from .matcher import build_matcher
from .utils import *
from .point_features import *


# TODO not working now !!!



class Mask2FormerLoss(nn.Module):
    def __init__(self,
                 num_classes=10,
                 eos_coef=0.1,
                 weight_cls=2.0,
                 weight_mask=5.0,
                 weight_dice=5.0,
                 losses=["labels", "masks"],
                 num_points=12544,
                 oversample_ratio=3.0,
                 importance_sample_ratio=0.75,
                 ignore_index=255,
                 **kwargs
                 ):

        super().__init__()
        self.num_classes = num_classes
        self.matcher = build_matcher(weight_cls, weight_mask, weight_dice, num_points)
        self.weight_dict = build_weight_dict(weight_cls, weight_mask, weight_dice)
        self.eos_coef = eos_coef
        self.losses = losses
        self.ignore_index = ignore_index

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio


    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses


    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]  #
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks)
        }

        del src_masks
        del target_masks
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)


    def forward(self, outputs, gt_masks):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             gt_masks: [bs, h_net_output, w_net_output]
        """
        targets = self._prepare_targets(gt_masks)

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        loss_cls, loss_mask, loss_dice = [], [], []
        for w_k, w_v in self.weight_dict.items():
            weighted_loss_value = w_v * losses[w_k]
            if 'loss_ce' in w_k:
                loss_cls.append(weighted_loss_value)
            elif 'loss_mask' in w_k:
                loss_mask.append(weighted_loss_value)
            elif 'loss_dice' in w_k:
                loss_dice.append(weighted_loss_value)
            else:
                raise NotImplementedError

        loss_cls, loss_mask, loss_dice = sum(loss_cls), sum(loss_mask), sum(loss_dice)
        losses_gather = {}
        losses_gather['loss_cls'] = loss_cls
        losses_gather['loss_mask'] = loss_mask
        losses_gather['loss_dice'] = loss_dice
        losses_gather['total_loss'] = loss_cls + loss_mask + loss_dice

        return losses_gather


    def _prepare_targets(self, gt_masks):
        targets = []
        for mask in gt_masks:
            classes = torch.unique(mask)
            classes = classes[classes != self.ignore_index]

            binary_masks = []
            for class_id in classes:
                binary_masks.append(mask == class_id)

            if len(binary_masks) == 0:
                final_masks = torch.zeros((0, mask.shape[-2], mask.shape[-1]))
            else:
                final_masks = torch.stack(binary_masks)

            targets.append({'masks': final_masks,
                            'labels': classes})

        return targets


    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)