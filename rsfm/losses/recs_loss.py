import torch
import torch.nn as nn
import torch.nn.functional as F
from .ris_loss import RISLoss
from .detr_based_od_loss.deformable_detr_loss import DeformableDETRLoss, get_world_size, is_dist_avail_and_initialized
from utils.bbox_utils import generalized_box_iou, xywh2xyxy



# class RIS2VGLoss(nn.Module):
#     def __init__(self,
#                  w_ris2vg_gt=1.0,
#                  w_ris2vg_vg=1.0,
#                  weight_bbox=5.0,
#                  weight_giou=2.0):
#         super().__init__()
#
#         self.w_ris2vg_gt = w_ris2vg_gt
#         self.w_ris2vg_vg = w_ris2vg_vg
#
#         self.weight_bbox = weight_bbox
#         self.weight_giou = weight_giou
#
#     def forward(self, inputs, gt_bboxs, vg_bboxs):
#         num_boxes = sum(1 for _ in range(gt_bboxs.shape[0]))
#         num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=inputs.device)
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(num_boxes)
#         num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
#
#         ris2vg_bboxs = self.mask2bbox(inputs)
#
#         loss_ris2vg_gt = self.bbox_loss(ris2vg_bboxs, gt_bboxs, num_boxes)
#         loss_ris2vg_vg = self.bbox_loss(ris2vg_bboxs, vg_bboxs['pred_boxes'].squeeze(1), num_boxes)
#
#         loss_ris2vg = loss_ris2vg_gt + loss_ris2vg_vg
#
#         return loss_ris2vg
#
#
#     def mask2bbox(self, inputs):
#         mask = (inputs.sigmoid() > 0.5).squeeze(1)
#         B, src_h, src_w = mask.shape
#         ris2vg_bboxs = torch.zeros((B, 4), device=mask.device, dtype=torch.float32)
#
#         for i in range(B):
#             foreobject = mask[i].nonzero(as_tuple=True)
#             if foreobject[0].numel() > 0:
#                 y_min, x_min = foreobject[0].min(), foreobject[1].min()
#                 y_max, x_max = foreobject[0].max(), foreobject[1].max()
#                 x_c = (x_max + x_min) / (2 * src_w)
#                 y_c = (y_max + y_min) / (2 * src_h)
#                 w = (x_max - x_min) / src_w
#                 h = (y_max - y_min) / src_h
#                 ris2vg_bboxs[i] = torch.tensor([x_c, y_c, w, h], device=mask.device, dtype=torch.float32)
#             else:
#                 ris2vg_bboxs[i] = torch.tensor([0., 0., 0., 0.], device=mask.device, dtype=torch.float32)
#
#         return torch.round(ris2vg_bboxs * 10000) / 10000
#
#
#     def bbox_loss(self, inputs, targets, num_boxes):
#         loss_bbox = F.l1_loss(inputs, targets, reduction='none')
#         loss_bbox = loss_bbox.sum() / num_boxes
#
#         loss_giou = 1 - torch.diag(generalized_box_iou(xywh2xyxy(inputs), xywh2xyxy(targets)))
#         loss_giou = loss_giou.sum() / num_boxes
#
#         loss = loss_bbox * self.weight_bbox + loss_giou * self.weight_giou
#
#         return loss
#
#
#
# class RECSLoss(nn.Module):
#     def __init__(self,
#                  weight_ris=1.0,
#                  weight_vg=0.1,
#                  # settings for ris
#                  weight_bce=1.0,
#                  weight_bdice=1.0,
#                  # settings for vg
#                  num_classes=1,
#                  weight_bbox=5.0,
#                  weight_giou=2.0,
#                  use_for_vg=True,
#                  **kwargs):
#         super().__init__()
#
#         self.weight_ris = weight_ris
#         self.loss_ris = RISLoss(weight_bce=weight_bce, weight_bdice=weight_bdice, **kwargs)
#
#         self.weight_vg = weight_vg
#         self.loss_vg = DeformableDETRLoss(num_classes=num_classes,
#                                           weight_cls=0.0,
#                                           weight_bbox=weight_bbox,
#                                           weight_giou=weight_giou,
#                                           use_for_vg=use_for_vg,
#                                           **kwargs)
#
#         self.loss_ris2vg = RIS2VGLoss()
#
#     def forward(self, outputs, masks, bboxs):
#         outputs_ris = outputs['mask']
#         outputs_vg = outputs['bbox']
#
#         loss_ris = self.weight_ris * self.loss_ris(outputs_ris, masks)
#         loss_vg = self.weight_vg * self.loss_vg(outputs_vg, bboxs)['total_loss']
#
#         final_loss = {}
#         final_loss['loss_ris'] = loss_ris
#         final_loss['loss_vg'] = loss_vg
#
#         # loss_ris2vg_gt = 0.1 * self.loss_ris2vg(outputs_ris, bboxs, outputs_vg)
#         # final_loss['loss_ris2vg_gt'] = loss_ris2vg_gt
#         # final_loss['total_loss'] = loss_ris + loss_vg + loss_ris2vg_gt
#
#         # loss_ris2vg_vg = 0.1 * self.loss_ris2vg(outputs_ris, bboxs, outputs_vg)
#         # final_loss['loss_ris2vg_vg'] = loss_ris2vg_vg
#         # final_loss['total_loss'] = loss_ris + loss_vg + loss_ris2vg_vg
#
#         loss_ris2vg = 0.01 * self.loss_ris2vg(outputs_ris, bboxs, outputs_vg)
#         final_loss['loss_ris2vg'] = loss_ris2vg
#         final_loss['total_loss'] = loss_ris + loss_vg + loss_ris2vg
#
#         return final_loss



class RECSLoss(nn.Module):
    def __init__(self,
                 weight_ris=1.0,
                 weight_vg=0.1,
                 # settings for ris
                 weight_bce=1.0,
                 weight_bdice=1.0,
                 # settings for vg
                 num_classes=1,
                 weight_bbox=5.0,
                 weight_giou=2.0,
                 use_for_vg=True,
                 **kwargs):
        super().__init__()

        self.weight_ris = weight_ris
        self.loss_ris = RISLoss(weight_bce=weight_bce, weight_bdice=weight_bdice, **kwargs)

        self.weight_vg = weight_vg
        self.loss_vg = DeformableDETRLoss(num_classes=num_classes,
                                          weight_cls=0.0,
                                          weight_bbox=weight_bbox,
                                          weight_giou=weight_giou,
                                          use_for_vg=use_for_vg,
                                          **kwargs)

    def forward(self, outputs, masks, bboxs):
        outputs_ris = outputs['mask']
        outputs_vg = outputs['bbox']

        loss_ris = self.weight_ris * self.loss_ris(outputs_ris, masks)
        loss_vg = self.weight_vg * self.loss_vg(outputs_vg, bboxs)['total_loss']

        final_loss = {}
        final_loss['loss_ris'] = loss_ris
        final_loss['loss_vg'] = loss_vg
        final_loss['total_loss'] = loss_ris + loss_vg

        return final_loss



# class RECSLoss(nn.Module):
#     def __init__(self,
#                  weight_ris=1.0,
#                  weight_vg=0.1,
#                  # settings for ris
#                  weight_bce=1.0,
#                  weight_bdice=1.0,
#                  # settings for vg
#                  num_classes=1,
#                  weight_bbox=5.0,
#                  weight_giou=2.0,
#                  use_for_vg=True,
#                  **kwargs):
#         super().__init__()
#
#         self.weight_ris = weight_ris
#         self.loss_ris = nn.CrossEntropyLoss()
#
#         self.weight_vg = weight_vg
#         self.loss_vg = DeformableDETRLoss(num_classes=num_classes,
#                                           weight_cls=0.0,
#                                           weight_bbox=weight_bbox,
#                                           weight_giou=weight_giou,
#                                           use_for_vg=use_for_vg,
#                                           **kwargs)
#
#     def forward(self, outputs, masks, bboxs):
#         outputs_ris = outputs['mask']
#         outputs_vg = outputs['bbox']
#
#         loss_ris = self.weight_ris * self.loss_ris(outputs_ris, masks)
#         loss_vg = self.weight_vg * self.loss_vg(outputs_vg, bboxs)['total_loss']
#
#         final_loss = {}
#         final_loss['loss_ris'] = loss_ris
#         final_loss['loss_vg'] = loss_vg
#         final_loss['total_loss'] = loss_ris + loss_vg
#
#         return final_loss



# from rsfm.losses.vg_loss import VGLoss
#
# class RECSLoss(nn.Module):
#     def __init__(self,
#                  weight_ris=1.0,
#                  weight_vg=1.0,
#                  weight_bbox=5.0,
#                  weight_giou=2.0,
#                  aux_loss=True,
#                  **kwargs):
#         super().__init__()
#
#         self.weight_ris = weight_ris
#         self.loss_ris = nn.CrossEntropyLoss()
#
#         self.weight_vg = weight_vg
#         self.loss_vg = VGLoss(weight_bbox, weight_giou, aux_loss)
#
#     def forward(self, outputs, masks, bboxs):
#         outputs_ris = outputs['mask']
#         outputs_vg = outputs['bbox']
#
#         loss_ris = self.weight_ris * self.loss_ris(outputs_ris, masks)
#         loss_vg = self.weight_vg * self.loss_vg(outputs_vg, bboxs)['total_loss']
#
#         final_loss = {}
#         final_loss['loss_ris'] = loss_ris
#         final_loss['loss_vg'] = loss_vg
#         final_loss['total_loss'] = loss_ris + loss_vg
#
#         return final_loss