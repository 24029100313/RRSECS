import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import PPM



class UPerHead(nn.Module):

    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 num_classes=7,
                 embedding_dim=512,
                 pool_scales=(1, 2, 3, 6),
                 norm_layer=nn.BatchNorm2d,
                 dropout_ratio=0.1,
                 align_corners=False,
                 **kwargs):
        super(UPerHead, self).__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.align_corners = align_corners
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.embedding_dim,
            norm_layer=norm_layer,
            align_corners=align_corners)
        self.bottleneck = nn.Sequential(
                nn.Conv2d(self.in_channels[-1] + len(pool_scales) * self.embedding_dim, self.embedding_dim, 3, padding=1),
                norm_layer(self.embedding_dim),
                nn.ReLU(inplace=True)
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.embedding_dim, 1),
                norm_layer(self.embedding_dim),
                nn.ReLU(inplace=False)
                )
            fpn_conv = nn.Sequential(
                nn.Conv2d(self.embedding_dim, self.embedding_dim, 3, padding=1),
                norm_layer(self.embedding_dim),
                nn.ReLU(inplace=False)
                )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = nn.Sequential(
                nn.Conv2d(len(self.in_channels) * self.embedding_dim, self.embedding_dim, 3, padding=1),
                norm_layer(self.embedding_dim),
                nn.ReLU(inplace=True)
                )
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(self.embedding_dim, num_classes, kernel_size=1)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.dropout(output)
        output = self.conv_seg(output)

        return output
