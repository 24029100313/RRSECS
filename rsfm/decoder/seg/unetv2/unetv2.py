from torch import nn
import torch.nn.functional as F
from torch import Tensor
from typing import List
from .utils import ChannelAttention, SpatialAttention, SDI, BasicConv2d



class UNetv2Head(nn.Module):
    """
    use SpatialAtt + ChannelAtt
    """
    def __init__(self,
                 in_channels=[32, 64, 160, 256],
                 in_index=[0, 1, 2, 3],
                 feature_strides=[4, 8, 16, 32],
                 embedding_dim=32,
                 num_classes=7,
                 deep_supervision=False,
                 **kwargs):
        super(UNetv2Head, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        self.in_channels = in_channels
        self.in_index = in_index
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.deep_supervision = deep_supervision

        self.ca_1 = ChannelAttention(c1_in_channels)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(c2_in_channels)
        self.sa_2 = SpatialAttention()

        self.ca_3 = ChannelAttention(c3_in_channels)
        self.sa_3 = SpatialAttention()

        self.ca_4 = ChannelAttention(c4_in_channels)
        self.sa_4 = SpatialAttention()

        self.Translayer_1 = BasicConv2d(c1_in_channels, embedding_dim, 1)
        self.Translayer_2 = BasicConv2d(c2_in_channels, embedding_dim, 1)
        self.Translayer_3 = BasicConv2d(c3_in_channels, embedding_dim, 1)
        self.Translayer_4 = BasicConv2d(c4_in_channels, embedding_dim, 1)

        self.sdi_1 = SDI(embedding_dim)
        self.sdi_2 = SDI(embedding_dim)
        self.sdi_3 = SDI(embedding_dim)
        self.sdi_4 = SDI(embedding_dim)

        self.seg_outs = nn.ModuleList([
            nn.Conv2d(embedding_dim, num_classes, 1, 1) for _ in range(4)])

        self.deconv2 = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=4, stride=2,
                                          padding=1, bias=False)

    def forward(self, inputs: List[Tensor]):
        seg_outs = []
        x = [inputs[i] for i in self.in_index]  # len=4, 1/4,1/8,1/16,1/32
        f1, f2, f3, f4 = x

        f1 = self.ca_1(f1) * f1
        f1 = self.sa_1(f1) * f1
        f1 = self.Translayer_1(f1)

        f2 = self.ca_2(f2) * f2
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2)

        f3 = self.ca_3(f3) * f3
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3)

        f4 = self.ca_4(f4) * f4
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4)

        f41 = self.sdi_4([f1, f2, f3, f4], f4)
        f31 = self.sdi_3([f1, f2, f3, f4], f3)
        f21 = self.sdi_2([f1, f2, f3, f4], f2)
        f11 = self.sdi_1([f1, f2, f3, f4], f1)

        seg_outs.append(self.seg_outs[0](f41))

        y = self.deconv2(f41) + f31
        seg_outs.append(self.seg_outs[1](y))

        y = self.deconv3(y) + f21
        seg_outs.append(self.seg_outs[2](y))

        y = self.deconv4(y) + f11
        seg_outs.append(self.seg_outs[3](y))

        for i, o in enumerate(seg_outs):
            seg_outs[i] = F.interpolate(o, scale_factor=self.feature_strides[::-1][i], mode='bilinear')

        if self.deep_supervision:
            return seg_outs[::-1]
        else:
            return seg_outs[-1]

