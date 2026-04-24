from .dice_loss import DiceLoss, MultiClassDiceLoss
from .unetformer_loss import UnetFormerLoss, CombinedLoss
from .lovasz_loss import LovaszLoss
from .ce_loss import SoftCrossEntropyLoss
from .jaccard_loss import JaccardLoss
from .bce_loss import SoftBCELoss
from .ris_loss import RISLoss
from .vg_loss import VGLoss, LQVGLoss
from .recs_loss import RECSLoss
from .detr_based_od_loss import *
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


__all__ = ['DiceLoss', 'UnetFormerLoss', 'CombinedLoss', 'LovaszLoss', 'SoftCrossEntropyLoss',
           'JaccardLoss', 'SoftBCELoss', 'RISLoss', 'VGLoss', 'LQVGLoss',
           'DETRLoss', 'DINOLoss', 'DeformableDETRLoss', 'Mask2FormerLoss', 'MultiClassDiceLoss',
           'LabelSmoothingCrossEntropy', 'SoftTargetCrossEntropy', 'RECSLoss']
