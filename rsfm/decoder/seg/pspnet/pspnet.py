import torch.nn as nn
from .utils import PSPModule, CBR



class PSPNetHead(nn.Module):
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 dropout_ratio=0.2,
                 num_classes=7,
                 embedding_dim=512,
                 **kwargs):
        super().__init__()

        self.psp = PSPModule(
            in_channels=in_channels[-1],
            sizes=(1, 2, 3, 6),
        )

        self.conv = CBR(in_channels[-1] * 2, embedding_dim, 3, 1, 1)

        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, inputs):
        x = inputs[-1]

        x = self.psp(x)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.conv_seg(x)

        return x