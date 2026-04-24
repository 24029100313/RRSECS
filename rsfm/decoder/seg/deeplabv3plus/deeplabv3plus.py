import torch
from torch import nn
import torch.nn.functional as F
from .utils import ASPPModule
from rsfm.module.neck.utils import CBR


class DeepLabV3PlusHead(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 embedding_dim=256,
                 dilations=[12, 24, 36],
                 num_classes=7,
                 **kwargs):
        super(DeepLabV3PlusHead, self).__init__()

        self.in_channels = in_channels
        low_channels, _, _, high_channels = self.in_channels

        self.head = ASPPModule(high_channels, dilations)

        self.reduce = CBR(low_channels, 48, 1)

        self.fuse = nn.Sequential(CBR(high_channels // 8 + 48, embedding_dim, 3, 1, 1),
                                  CBR(embedding_dim, embedding_dim, 3, 1, 1))

        self.classifier = nn.Conv2d(embedding_dim, num_classes, 1, bias=True)

    def forward(self, inputs):
        c1, c4 = inputs[0], inputs[-1]
        out = self._decode(c1, c4)

        return out

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        out = self.classifier(feature)

        return out
