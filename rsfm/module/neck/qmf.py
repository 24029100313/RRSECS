from torch import nn
import torch.nn.functional as F
from .utils import CBR, SoftMuPatchMerging



class QMF(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs=5, in_index=[0, 1, 2, 3]):
        super().__init__()
        self.in_channels = in_channels
        self.num_outs = num_outs
        self.in_index = in_index

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.channel_align_layers = nn.ModuleList([
            CBR(c1_in_channels, out_channels),  # 0
            CBR(c2_in_channels, out_channels),  # 1
            CBR(c3_in_channels, out_channels),  # 2
            CBR(c4_in_channels, out_channels),  # 3
        ])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.fpn_down = nn.ModuleList()
        for i in range(len(self.channel_align_layers) - 1, 0, -1):
            self.fpn_down.append(SoftMuPatchMerging(out_channels))

    def forward(self, inputs, text):
        assert len(inputs) == len(self.in_channels)

        pyramid_features = [self.channel_align_layers[i](inputs[i]) for i in self.in_index]

        used_backbone_levels = len(pyramid_features)

        for i in range(len(self.in_index) - 2, -1, -1):
            pyramid_features[i] = pyramid_features[i] + self.upsample(pyramid_features[i + 1])

        outs = [pyramid_features[i] for i in range(used_backbone_levels)]

        for i in range(used_backbone_levels - 1):
            gated_feat = self.fpn_down[i](outs[i], text)
            outs[i + 1] = outs[i + 1] + gated_feat

        if self.num_outs > len(outs):
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))

        return outs