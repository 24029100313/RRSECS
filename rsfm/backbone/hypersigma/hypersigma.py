from .spatial_vit import *
from .spectral_vit import *
import torch.nn as nn


class DoubleConv_pad(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_sz):
        super().__init__()
        self.pad = 1 if kernel_sz == 3 else 0
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=(kernel_sz, kernel_sz), padding=self.pad),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class HyperSIGMA(nn.Module):
    def __init__(self, img_size=224, in_channels=3, arch='base', **kwargs):
        super().__init__()

        spat_arch_dict = {'base': (vit_base_patch16_spatsigma(img_size=img_size, in_channels=in_channels), 768),
                          'large': (vit_large_patch16_spatsigma(img_size=img_size, in_channels=in_channels), 1024),
                          'huge': (vit_huge_patch16_spatsigma(img_size=img_size, in_channels=in_channels), 1280)}

        spec_arch_dict = {'base': vit_base_patch16_specsigma(img_size=img_size, in_channels=in_channels),
                          'large': vit_large_patch16_specsigma(img_size=img_size, in_channels=in_channels),
                          'huge': vit_huge_patch16_specsigma(img_size=img_size, in_channels=in_channels)}

        self.spat_encoder, embed_dim = spat_arch_dict[arch]
        self.spec_encoder = spec_arch_dict[arch]
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_spec = nn.Sequential(
            nn.Linear(100, 768, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(768, 768, bias=False),
            nn.Sigmoid(),
        )

    def init_weights(self, pretrained=None):
        spat_pretrained, spec_pretrained = pretrained
        self.spat_encoder.init_weights(spat_pretrained)
        self.spec_encoder.init_weights(spec_pretrained)

    def forward(self, x):
        b, _, h, w = x.shape
        spat_features = self.spat_encoder(x)

        x_spe = self.spec_encoder(x)[0]
        x_spe = self.pool(x_spe).view(b, -1)
        x_spe_weights = self.fc_spec(x_spe).view(b, -1, 1, 1)

        fused_features = []
        for spat_feature in spat_features:
            fused_features.append((1 + x_spe_weights) * spat_feature)

        return fused_features


class vit_base_patch16_hypersigma(HyperSIGMA):
    def __init__(self, **kwargs):
        super(vit_base_patch16_hypersigma, self).__init__(arch='base', **kwargs)


class vit_large_patch16_hypersigma(HyperSIGMA):
    def __init__(self, **kwargs):
        super(vit_large_patch16_hypersigma, self).__init__(arch='large', **kwargs)


class vit_huge_patch16_hypersigma(HyperSIGMA):
    def __init__(self, **kwargs):
        super(vit_huge_patch16_hypersigma, self).__init__(arch='huge', **kwargs)