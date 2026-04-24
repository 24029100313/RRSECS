import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from rsfm.module.vg_fusion.qrnet_fusion.utils import QueryDynamicAttention



class CBR(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 with_residual=False):
        super(CBR, self).__init__()
        self.with_residual = with_residual

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        if not self.with_residual:
            out = self.act(self.norm(self.conv(x)))
        else:
            out = x + self.act(self.norm(self.conv(x)))

        return out


class PBR(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 type='avg'):
        super(PBR, self).__init__()
        assert type in ['avg', 'max']

        if type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding)
        else:
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)

        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.act(self.norm(self.pool(x)))

        return out


def get_padding4dilation(k, d):
    p = (d * (k - 1)) // 2

    return p


class CGR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(CGR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.GroupNorm(32, out_channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.act(self.norm(self.conv(x)))

        return out


class SoftMuPatchMerging(nn.Module):
    def __init__(self, dim, use_spatial=True, use_channel=True):
        super(SoftMuPatchMerging, self).__init__()

        self.qdatt = QueryDynamicAttention(gate_channels=dim,
                                           mu_dim=768,
                                           reduction_ratio=16,
                                           pool_types=['avg', 'max'],
                                           use_spatial=use_spatial,
                                           use_channel=use_channel)

    def forward(self, x, text):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H, W, C).
            H, W: Spatial resolution of the input feature.
        """
        x = x.permute(0, 2, 3, 1)  # B,C,H,W to B,H,W,C
        B, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = rearrange(x, 'B H W C -> B (H W) C')

        x = self.qdatt(x, text)

        x = rearrange(x, 'B (H W) C -> B H W C', H=H, W=W)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.mean(torch.stack([x0, x1, x2, x3], -1), -1)  # B H/2 W/2 C

        x = x.permute(0, 3, 1, 2)

        return x