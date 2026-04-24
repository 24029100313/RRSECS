import torch.nn as nn
from .utils import DCFAM, Conv
from rsfm.module.neck.utils import CBR


class DCSwinHead(nn.Module):
    '''
    from 'A Novel Transformer Based Semantic Segmentation Scheme for Fine-Resolution Remote Sensing Images, GRSL 2022'
    '''
    def __init__(self,
                 num_classes=7,
                 in_channels=[96, 192, 384, 768],
                 embedding_dim=96,
                 dropout=0.05,
                 atrous_rates=(6, 12),
                 **kwargs):
        super(DCSwinHead, self).__init__()

        self.dcfam = DCFAM(in_channels, atrous_rates)

        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

        self.segmentation_head = nn.Sequential(
            CBR(in_channels[0], in_channels[0], 3, 1, 1),
            Conv(in_channels[0], num_classes, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=4))

        self.up = nn.Sequential(
            CBR(in_channels[1], in_channels[0], 3, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=2)
        )


    def forward(self, inputs):
        x1, x2, x3, x4 = inputs

        out1, out2 = self.dcfam(x1, x2, x3, x4)
        x = out1 + self.up(out2)
        x = self.dropout(x)
        x = self.segmentation_head(x)

        return x
