import torch
from torch import nn
from mmcv.cnn import ConvModule
from rsfm.utils import resize
from .utils import Hamburger



class SLViTHead(nn.Module):
    def __init__(self,
                 in_channels,
                 ham_norm_cfg,
                 num_classes=2,
                 embedding_dim=512,
                 dropout_ratio=0.,
                 ham_channels=512,
                 ham_kwargs=dict()):
        super(SLViTHead, self).__init__()

        self.ham_channels = ham_channels
        self.in_channels = in_channels
        self.align_corners = False

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            norm_cfg=ham_norm_cfg)

        self.align = ConvModule(
            self.ham_channels,
            in_channels[-2],
            1,
            norm_cfg=ham_norm_cfg)

        self.hamburger = Hamburger(self.ham_channels, ham_kwargs)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.conv_seg = nn.Conv2d(in_channels[-2], num_classes, 1)

    def forward(self, inputs, l):
        inputs = [resize(level, size=inputs[0].shape[2:],
                         mode='bilinear', align_corners=self.align_corners)
                  for level in inputs]

        x = torch.cat(inputs, dim=1)
        x = self.squeeze(x)
        x = self.hamburger(x, l)

        # align
        x = self.align(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return self.conv_seg(x)