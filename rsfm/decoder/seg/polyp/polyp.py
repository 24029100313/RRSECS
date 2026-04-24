import torch.nn as nn
import torch.nn.functional as F
from rsfm.module.neck.utils import CBR
from .utils import CFM, SAM, ChannelAttention, SpatialAttention


class PolypHead(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 embedding_dim=32,
                 num_classes=1,
                 **kwargs):
        super(PolypHead, self).__init__()

        self.Translayer2_0 = CBR(in_channels[0], embedding_dim, 1)
        self.Translayer2_1 = CBR(in_channels[1], embedding_dim, 1)
        self.Translayer3_1 = CBR(in_channels[2], embedding_dim, 1)
        self.Translayer4_1 = CBR(in_channels[3], embedding_dim, 1)

        self.CFM = CFM(embedding_dim)
        self.ca = ChannelAttention(in_channels[0])
        self.sa = SpatialAttention()
        self.SAM = SAM()

        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.out_SAM = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, inputs):
        x1, x2, x3, x4 = inputs

        # CIM
        x1 = self.ca(x1) * x1  # channel attention
        cim_feature = self.sa(x1) * x1  # spatial attention

        # CFM
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)

        cfm_feature = self.CFM(x4_t, x3_t, x2_t)

        # SAM
        T2 = self.Translayer2_0(cim_feature)
        T2 = self.down05(T2)
        sam_feature = self.SAM(cfm_feature, T2)

        out = self.out_SAM(sam_feature)
        out = F.interpolate(out, scale_factor=8, mode='bilinear')

        return out