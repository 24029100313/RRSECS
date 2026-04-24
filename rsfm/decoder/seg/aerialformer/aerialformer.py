import torch
import torch.nn as nn
from .utils import MDCBlock, up_pooling
from rsfm.module.neck.utils import CBR


class AerialFormerHead(nn.Module):
    """
    from 'AerialFormer: Multi-resolution Transformer for Aerial Image Segmentation, RS 2024'
    """
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 num_classes=7,
                 embedding_dim=96,
                 in_index=[0, 1, 2, 3],
                 dropout=0.1,
                 **kwargs):
        super(AerialFormerHead, self).__init__()

        self.embedding_dim = in_channels[0]
        self.in_index = in_index

        self.in_channels = in_channels[::-1]

        self.up_convs = nn.ModuleList()
        self.dilated_convs = nn.ModuleList()

        self.dropout = nn.Dropout2d(dropout)
        self.conv_seg = nn.Conv2d(self.embedding_dim, num_classes, kernel_size=1)

        custom_params_list = [
            {
                # Deepest Layer
                "kernel": (3, 3, 3),
                "padding": (1, 2, 3),
                "dilation": (1, 2, 3),
            },
            {
                "kernel": (3, 3, 3),
                "padding": (1, 2, 3),
                "dilation": (1, 2, 3),
            },
            {
                "kernel": (3, 3, 3),
                "padding": (1, 2, 3),
                "dilation": (1, 2, 3),
            },
            {
                "kernel": (3, 3, 3),
                "padding": (1, 1, 1),
                "dilation": (1, 1, 1),
            }
        ]

        for idx in range(len(self.in_channels)):
            if idx != 0:
                self.up_convs.append(
                    up_pooling(self.in_channels[idx - 1], self.in_channels[idx])
                )
            else:
                self.up_convs.append(nn.Identity())

            self.dilated_convs.append(
                nn.Sequential(
                    MDCBlock(
                        in_channels=self.in_channels[idx] * 2 ** (idx != 0),
                        out_channels=self.in_channels[idx],
                        custom_params=custom_params_list[idx],
                    ),
                    CBR(self.in_channels[idx], self.in_channels[idx], 3, 1, 1)
                )
            )

    def forward(self, inputs):
        inputs = inputs[::-1]

        x = inputs[0]
        x = self.dilated_convs[0](x)

        for idx in range(1, len(inputs)):
            x = self.up_convs[idx](x)
            x = torch.cat([x, inputs[idx]], dim=1)
            x = self.dilated_convs[idx](x)

        output = self.dropout(x)
        out = self.conv_seg(output)

        return out
