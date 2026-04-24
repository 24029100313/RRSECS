import torch
from torch import nn
from rsfm.module.conv.dysample import dysample_splus
from .utils import CLM, CBR



# v1
class MCTHead(nn.Module):
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 embedding_dim=512,
                 l_dim=768,
                 num_classes=2,
                 **kwargs):
        super(MCTHead, self).__init__()

        self.clm4 = CLM(in_channels[3], l_dim, 8)
        self.conv4 = CBR(in_channels[3], embedding_dim, 3, 1, 1)
        self.up4 = dysample_splus(in_channels=embedding_dim, groups=4)

        self.mix43 = CBR(in_channels[2] + embedding_dim, embedding_dim, 1)
        self.clm43 = CLM(embedding_dim, l_dim, 8)
        self.conv43 = CBR(embedding_dim, embedding_dim, 3, 1, 1)
        self.up43 = dysample_splus(in_channels=embedding_dim, groups=4)

        self.mix432 = CBR(in_channels[1] + embedding_dim, embedding_dim, 1)
        self.clm432 = CLM(embedding_dim, l_dim, 8)
        self.conv432 = CBR(embedding_dim, embedding_dim, 3, 1, 1)
        self.up432 = dysample_splus(in_channels=embedding_dim, groups=4)

        self.mix4321 = CBR(in_channels[0] + embedding_dim, embedding_dim, 1)
        self.clm4321 = CLM(embedding_dim, l_dim, 8)
        self.conv4321 = CBR(embedding_dim, embedding_dim, 3, 1, 1)
        self.up4321 = dysample_splus(in_channels=embedding_dim, groups=4)

        self.conv_seg = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, inputs, l):
        x1, x2, x3, x4 = inputs

        x4 = self.clm4(x4, l)
        x4 = self.conv4(x4)
        x4 = self.up4(x4)

        x43 = torch.cat((x4, x3), dim=1)

        x43 = self.mix43(x43)
        x43 = self.clm43(x43, l)
        x43 = self.conv43(x43)
        x43 = self.up43(x43)

        x432 = torch.cat((x43, x2), dim=1)

        x432 = self.mix432(x432)
        x432 = self.clm432(x432, l)
        x432 = self.conv432(x432)
        x432 = self.up432(x432)

        x4321 = torch.cat((x432, x1), dim=1)

        x4321 = self.mix4321(x4321)
        x4321 = self.clm4321(x4321, l)
        x4321 = self.conv4321(x4321)
        x4321 = self.up4321(x4321)

        out = self.conv_seg(x4321)

        return out



# self.conv_seg = nn.Sequential(CBR(embedding_dim, embedding_dim, 3, 1, 1),
#                               dysample_splus(in_channels=embedding_dim, groups=4),
#                               nn.Conv2d(embedding_dim, num_classes, 1))
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 85.11 | 80.27 | 71.98 | 57.09 |  24.4 | 84.55 | 74.05 |
# +-------+-------+-------+-------+-------+-------+-------+



# v2
# from rsfm.module.neck.mscab import MSCAB
#
# class MCTHead(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  embedding_dim=512,
#                  l_dim=768,
#                  num_classes=2):
#         super(MCTHead, self).__init__()
#
#         self.mscab = MSCAB(img_size=512,
#                            in_channels=in_channels,
#                            out_channels=embedding_dim,
#                            l_dim=l_dim)
#
#         # Cross-scale fusion module
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.csf = nn.ModuleList([
#             nn.Sequential(CBR(2 * embedding_dim, embedding_dim, 1),
#                           CBR(embedding_dim, embedding_dim, 3, 1, 1)) for _ in range(len(in_channels) - 1)
#         ])
#
#         self.conv_seg = nn.Conv2d(embedding_dim, num_classes, 1)
#
#     def forward(self, inputs, l):
#         vl_feats = self.mscab(inputs, l)
#         vl_feats = vl_feats[::-1]
#
#         for i in range(len(vl_feats) - 1):
#             vl_feats[i + 1] = self.csf[i](torch.cat((self.upsample(vl_feats[i]), vl_feats[i + 1]), dim=1))
#
#         out = self.conv_seg(vl_feats[-1])
#
#         return out