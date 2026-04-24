from .joint_loss import JointLoss
from torch import nn, Tensor
from .dice_loss import DiceLoss
from .ce_loss import SoftCrossEntropyLoss



class UnetFormerLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                              DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)

        return loss


class CombinedLoss(nn.Module):

    def __init__(self, ignore_index=255, weight_ce=1.0, weight_dice=1.0, smooth_factor=0.):
        super().__init__()
        self.loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=smooth_factor, ignore_index=ignore_index),
                              DiceLoss(smooth=smooth_factor, ignore_index=ignore_index),
                              weight_ce,
                              weight_dice)

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)

        return loss
