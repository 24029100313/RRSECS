import torch.nn as nn
from rsfm.decoder import DPTHead
import torch.nn.functional as F
from rsfm.backbone.dinov2 import dinov2_small, dinov2_base, dinov2_large



def get_encoder(type, img_size):
    encs = {'small': {'enc': dinov2_small(img_size=img_size), 'embed_dim': 384, 'features': 64},
            'base': {'enc': dinov2_base(img_size=img_size),  'embed_dim': 768, 'features': 128},
            'large': {'enc': dinov2_large(img_size=img_size), 'embed_dim': 1024, 'features': 256}}

    return encs[type]['enc'], encs[type]['embed_dim'], encs[type]['features']


class DPT(DPTHead):
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 embedding_dim=96,
                 use_bn=False,
                 use_depth=True):
        super(DPT, self).__init__(in_channels=in_channels,
                                  embedding_dim=embedding_dim,
                                  use_bn=use_bn,
                                  use_depth=use_depth)

        out_channels = in_channels

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels[-1], out_channel, 1, 1, 0) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, 4, 0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, 2, 0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, 2, 1)
        ])

    def forward(self, inputs):
        outs = []
        for i, x in enumerate(inputs):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            outs.append(x)

        layer_1, layer_2, layer_3, layer_4 = outs

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if not self.use_depth:
            out = self.scratch.output_conv(path_1)
        else:
            out = self.scratch.output_conv1(path_1)
            out = F.interpolate(out, (int(layer_3.shape[2] * 14), int(layer_3.shape[3] * 14)),
                                mode="bilinear", align_corners=True)
            out = self.scratch.output_conv2(out)

        return out