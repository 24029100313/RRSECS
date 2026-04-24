import torch.nn as nn
from .utils import CBR



class FPN(nn.Module):
    def __init__(self, in_channels, out_channels, in_index=[0, 1, 2, 3]):
        super(FPN, self).__init__()
        self.in_index = in_index
        self.in_channels = in_channels

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.channel_align_layers = nn.ModuleList([
            CBR(c1_in_channels, out_channels), #0
            CBR(c2_in_channels, out_channels), #1
            CBR(c3_in_channels, out_channels), #2
            CBR(c4_in_channels, out_channels), #3
        ])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.fpn_layers = nn.ModuleList([
            CBR(out_channels, out_channels, 3, 1, 1),  # 0
            CBR(out_channels, out_channels, 3, 1, 1),  # 1
            CBR(out_channels, out_channels, 3, 1, 1),  # 2
            CBR(out_channels, out_channels, 3, 1, 1),  # 3
        ])

    def forward(self, inputs):
        pyramid_features = [self.channel_align_layers[i](inputs[i]) for i in self.in_index]
        for i in range(len(self.in_index) - 2, -1, -1):
            pyramid_features[i] = pyramid_features[i] + self.upsample(pyramid_features[i + 1])
        outs = [self.fpn_layers[i](pyramid_features[i]) for i in self.in_index]

        return outs