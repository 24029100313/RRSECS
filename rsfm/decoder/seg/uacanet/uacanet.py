import torch
import torch.nn as nn
from .utils import UACA, PAA_e, PAA_d


class UACANetHead(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 num_classes=7,
                 embedding_dim=256,
                 **kwargs):
        super(UACANetHead, self).__init__()

        self.context2 = PAA_e(in_channels[1], embedding_dim)
        self.context3 = PAA_e(in_channels[2], embedding_dim)
        self.context4 = PAA_e(in_channels[3], embedding_dim)

        self.decoder = PAA_d(embedding_dim)

        self.attention2 = UACA(embedding_dim * 2, embedding_dim)
        self.attention3 = UACA(embedding_dim * 2, embedding_dim)
        self.attention4 = UACA(embedding_dim * 2, embedding_dim)

        self.conv_seg = nn.Conv2d(1, num_classes, kernel_size=1)

    def forward(self, inputs):
        x1, x2, x3, x4 = inputs

        x2 = self.context2(x2)
        x3 = self.context3(x3)
        x4 = self.context4(x4)

        f5, a5 = self.decoder(x4, x3, x2)

        f4, a4 = self.attention4(torch.cat([x4, self.ret(f5, x4)], dim=1), a5)

        f3, a3 = self.attention3(torch.cat([x3, self.ret(f4, x3)], dim=1), a4)

        _, a2 = self.attention2(torch.cat([x2, self.ret(f3, x2)], dim=1), a3)

        out = self.conv_seg(a2)

        return out