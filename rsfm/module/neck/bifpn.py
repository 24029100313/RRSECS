from torch import nn
from .utils import CBR



class BiFPN(nn.Module):
    def __init__(self, in_channels, out_channels, in_index=[0, 1, 2, 3]):
        super(BiFPN, self).__init__()
        self.in_index = in_index
        self.in_channels = in_channels

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.channel_align_layers = nn.ModuleList([
            CBR(c1_in_channels, out_channels),  # 0
            CBR(c2_in_channels, out_channels),  # 1
            CBR(c3_in_channels, out_channels),  # 2
            CBR(c4_in_channels, out_channels),  # 3
        ])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.c3br_2 = CBR(out_channels, out_channels, 3, 1, 1)
        self.c3br_1 = CBR(out_channels, out_channels, 3, 1, 1)

        self.ds_0 = CBR(out_channels, out_channels, 3, 2, 1)
        self.ds_1 = CBR(out_channels, out_channels, 3, 2, 1)
        self.ds_2 = CBR(out_channels, out_channels, 3, 2, 1)

        self.bifpn_layers = nn.ModuleList([
            CBR(out_channels, out_channels, 3, 1, 1),  # 0
            CBR(out_channels, out_channels, 3, 1, 1),  # 1
            CBR(out_channels, out_channels, 3, 1, 1),  # 2
            CBR(out_channels, out_channels, 3, 1, 1),  # 3
        ])

    def forward(self, inputs):
        pyramid_features = [self.channel_align_layers[i](inputs[i]) for i in self.in_index]
        m0, m1, m2, m3 = pyramid_features

        m2_3 = m2 + self.upsample(m3)
        m1_2 = m1 + self.upsample(m2_3)

        f0 = m0 + self.upsample(m1_2)
        f1 = self.ds_0(f0) + m1 + self.c3br_1(m1_2)
        f2 = self.ds_1(f1) + m2 + self.c3br_2(m2_3)
        f3 = self.ds_2(f2) + m3

        bifpn_feats = [f0, f1, f2, f3]
        outs = [self.bifpn_layers[i](bifpn_feats[i]) for i in self.in_index]

        return outs