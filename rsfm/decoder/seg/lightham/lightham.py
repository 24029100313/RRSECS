import torch
import torch.nn as nn
from .utils import Hamburger
from mmcv.cnn import ConvModule
from rsfm.utils import resize



class LightHamHead(nn.Module):
    """
    Is Attention Better Than Matrix Decomposition?
    """

    def __init__(self,
                 in_channels,
                 num_classes=7,
                 embedding_dim=256,
                 ham_channels=256,
                 ham_kwargs=dict(),
                 dropout_ratio=0.1,
                 **kwargs):
        super(LightHamHead, self).__init__()
        self.in_channels = in_channels
        self.ham_channels = ham_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            norm_cfg=dict(type='SyncBN', requires_grad=True))

        self.hamburger = Hamburger(ham_channels, ham_kwargs)

        self.align = ConvModule(
            self.ham_channels,
            embedding_dim,
            1,
            norm_cfg=dict(type='SyncBN', requires_grad=True))

        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, inputs):
        inputs = [resize(
            level,
            size=inputs[0].shape[2:],
            mode='bilinear',
            align_corners=False
        ) for level in inputs]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        output = self.align(x)

        output = self.dropout(output)
        output = self.conv_seg(output)

        return output