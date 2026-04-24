import torch
from torch import nn
import torch.nn.functional as F
from utils.bbox_utils import xywh2xyxy, generalized_box_iou
from einops import rearrange



class l1_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label):
        B = pred.shape[0]
        loss = F.l1_loss(pred, label, reduction='none')
        loss = loss.sum() / B

        return loss


class giou_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label):
        B = pred.shape[0]
        loss = 1 - torch.diag(generalized_box_iou(xywh2xyxy(pred), xywh2xyxy(label)))
        loss = loss.sum() / B

        return loss


class VGLoss(nn.Module):
    def __init__(self, weight_bbox=5.0, weight_giou=2.0, aux_loss=False):
        super().__init__()

        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.aux_loss = aux_loss

        self.loss_bbox = l1_loss()
        self.loss_giou = giou_loss()

    def forward(self, logits, labels):
        losses = {}

        if self.aux_loss:
            loss_bbox, loss_giou = [], []
            for logit in logits:
                loss_bbox.append(self.loss_bbox(logit, labels))
                loss_giou.append(self.loss_giou(logit, labels))

            loss_bbox = sum(loss_bbox) * self.weight_bbox
            loss_giou = sum(loss_giou) * self.weight_giou

            losses['loss_bbox'] = loss_bbox
            losses['loss_giou'] = loss_giou
            losses['total_loss'] = loss_bbox + loss_giou

            return losses

        loss_bbox = self.loss_bbox(logits, labels)
        loss_giou = self.loss_giou(logits, labels)

        losses['loss_bbox'] = loss_bbox * self.weight_bbox
        losses['loss_giou'] = loss_giou * self.weight_giou
        losses['total_loss'] = loss_bbox * self.weight_bbox + loss_giou * self.weight_giou

        return losses


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 2, cost_bbox: float = 5, cost_giou: float = 2):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
            cost_mask: This is the relative weight of the sigmoid focal loss of the mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 , "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries_per_frame, num_frames, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries_per_frame, num_frames, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 NOTE: Since every frame has one object at most
                 "labels": Tensor of dim [num_frames] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_frames, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        src_logits = outputs["pred_logits"]
        src_boxes = outputs["pred_boxes"]

        bs, nf, nq, _ = src_boxes.shape

        indices = []
        for i in range(bs):
            out_prob = src_logits[i].sigmoid()
            out_bbox = src_boxes[i]

            tgt_bbox = targets[i].unsqueeze(0)

            # class cost
            cost_class = []
            for t in range(nf):
                out_prob_split = out_prob[t]

                # Compute the classification cost.
                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * (out_prob_split ** gamma) * (-(1 - out_prob_split + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob_split) ** gamma) * (-(out_prob_split + 1e-8).log())

                cost_class_split = pos_cost_class[:, [0]] - neg_cost_class[:, [0]]
                cost_class.append(cost_class_split)
            cost_class = torch.stack(cost_class, dim=0).mean(0)  # [q, 1]

            # box cost
            cost_bbox, cost_giou = [], []
            for t in range(nf):
                out_bbox_split = out_bbox[t]
                tgt_bbox_split = tgt_bbox[t].unsqueeze(0)

                # Compute the L1 cost between boxes
                cost_bbox_split = torch.cdist(out_bbox_split, tgt_bbox_split, p=1)

                # Compute the giou cost betwen boxes
                cost_giou_split = -generalized_box_iou(xywh2xyxy(out_bbox_split),
                                                       xywh2xyxy(tgt_bbox_split))

                cost_bbox.append(cost_bbox_split)
                cost_giou.append(cost_giou_split)

            cost_bbox = torch.stack(cost_bbox, dim=0).mean(0)
            cost_giou = torch.stack(cost_giou, dim=0).mean(0)

            # Final cost matrix
            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou  # [q, 1]

            # Only has one tgt, MinCost Matcher
            _, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(1).to(src_ind)
            indices.append((src_ind.long(), tgt_ind.long()))

        return indices


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class LQVGLoss(nn.Module):
    def __init__(self, weight_cls=2.0, weight_bbox=5.0, weight_giou=2.0, aux_loss=False, num_classes=1):
        super().__init__()

        self.weight_cls = weight_cls
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.num_classes = num_classes

        self.matcher = HungarianMatcher(cost_class=weight_cls,
                                        cost_bbox=weight_bbox,
                                        cost_giou=weight_giou)

        self.aux_loss = aux_loss


    def loss_labels(self, outputs, targets, indices):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        num_boxes, nf, nq = src_logits.shape[:3]
        src_logits = rearrange(src_logits, 'b t q k -> b (t q) k')

        # judge the valid frames
        valid_indices = []
        valids = [torch.tensor([1], device=targets.device) for _ in range(targets.shape[0])]
        for valid, (indice_i, indice_j) in zip(valids, indices):
            valid_ind = valid.nonzero().flatten()
            valid_i = valid_ind * nq + indice_i
            valid_j = valid_ind + indice_j * nf
            valid_indices.append((valid_i, valid_j))

        idx = self._get_src_permutation_idx(valid_indices) # NOTE: use valid indices
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = 0

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=0.25, gamma=2) * src_logits.shape[1]

        return loss_ce


    def loss_boxes(self, outputs, targets, indices):
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes'] # [B, 1, Q, 4]
        num_boxes = src_boxes.shape[0]
        src_boxes = src_boxes.transpose(1, 2) # [B, Q, 1, 4]

        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx] # [B, 1, 4]
        src_boxes = src_boxes.flatten(0, 1)  # [b*t, 4]

        loss_bbox = F.l1_loss(src_boxes, targets, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(xywh2xyxy(src_boxes),
                                                       xywh2xyxy(targets)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


    def forward(self, logits, labels):
        losses = {}

        if self.aux_loss:
            loss_cls, loss_bbox, loss_giou = [], [], []
            for outputs in logits:
                indices = self.matcher(outputs, labels)
                loss_cls.append(self.loss_labels(outputs, labels, indices))
                losses_box = self.loss_boxes(outputs, labels, indices)
                loss_bbox.append(losses_box['loss_bbox'])
                loss_giou.append(losses_box['loss_giou'])

            loss_cls = sum(loss_cls)
            loss_bbox = sum(loss_bbox)
            loss_giou = sum(loss_giou)

            losses['loss_cls'] = loss_cls
            losses['loss_bbox'] = loss_bbox
            losses['loss_giou'] = loss_giou
            losses['total_loss'] = loss_cls * self.weight_cls + \
                                   loss_bbox * self.weight_bbox + \
                                   loss_giou * self.weight_giou

            return losses

        indices = self.matcher(logits, labels)
        loss_cls = self.loss_labels(logits, labels, indices)
        losses_box = self.loss_boxes(logits, labels, indices)
        loss_bbox = losses_box['loss_bbox']
        loss_giou = losses_box['loss_giou']

        losses['loss_cls'] = loss_cls
        losses['loss_bbox'] = loss_bbox
        losses['loss_giou'] = loss_giou
        losses['total_loss'] = loss_cls * self.weight_cls + \
                               loss_bbox * self.weight_bbox + \
                               loss_giou * self.weight_giou

        return losses
