from torch import nn
from .utils import CBR
from .fpn import FPN


class PAFPN(FPN):
    def __init__(self, in_channels, out_channels, in_index=[0, 1, 2, 3]):
        super(PAFPN, self).__init__(in_channels, out_channels)

        self.in_index = in_index
        self.in_channels = in_channels

        self.downsample_layers = nn.ModuleList([
            CBR(out_channels, out_channels, 3, 2, 1), #0
            CBR(out_channels, out_channels, 3, 2, 1), #1
            CBR(out_channels, out_channels, 3, 2, 1), #2
        ])

        self.pafpn_layers = nn.ModuleList([
            CBR(out_channels, out_channels, 3, 1, 1), #0
            CBR(out_channels, out_channels, 3, 1, 1), #1
            CBR(out_channels, out_channels, 3, 1, 1), #2
        ])

    def forward(self, inputs):
        pyramid_features = [self.channel_align_layers[i](inputs[i]) for i in self.in_index]
        for i in range(len(self.in_index) - 2, -1, -1):
            pyramid_features[i] = pyramid_features[i] + self.upsample(pyramid_features[i + 1])
        inter_outs = [self.fpn_layers[i](pyramid_features[i]) for i in self.in_index]

        for i in range(0, 3):
            inter_outs[i + 1] = inter_outs[i + 1] + self.downsample_layers[i](inter_outs[i])

        outs = []
        outs.append(inter_outs[0])
        outs.extend([self.pafpn_layers[i - 1](inter_outs[i]) for i in range(1, 4)])

        return outs