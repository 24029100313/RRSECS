import torch.nn as nn
from rsfm.decoder.seg.uacanet.utils import RFB, PPD, reverse_attention


class PraNetHead(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 num_classes=7,
                 embedding_dim=256,
                 **kwargs):
        super(PraNetHead, self).__init__()

        self.context2 = RFB(in_channels[1], embedding_dim)
        self.context3 = RFB(in_channels[2], embedding_dim)
        self.context4 = RFB(in_channels[3], embedding_dim)

        self.decoder = PPD(embedding_dim)

        self.attention2 = reverse_attention(in_channels[1], embedding_dim // 4, 2, 3)
        self.attention3 = reverse_attention(in_channels[2], embedding_dim // 4, 2, 3)
        self.attention4 = reverse_attention(in_channels[3], embedding_dim, 3, 5)

        self.conv_seg = nn.Conv2d(1, num_classes, kernel_size=1)

    def forward(self, inputs):
        x1, x2, x3, x4 = inputs

        x2_context = self.context2(x2)
        x3_context = self.context3(x3)
        x4_context = self.context4(x4)

        _, a5 = self.decoder(x4_context, x3_context, x2_context)
        _, a4 = self.attention4(x4, a5)
        _, a3 = self.attention3(x3, a4)
        _, a2 = self.attention2(x2, a3)

        out = self.conv_seg(a2)

        return out