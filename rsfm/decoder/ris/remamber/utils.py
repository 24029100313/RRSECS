import torch
from torch import nn
import torch.nn.functional as F
from rsfm.module.ris_fusion.remamber_fusion.utils import Linear2d



class UpSample2D(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()

        if not channel_first:
            raise
        self.proj = Linear2d(dim, dim_out, bias=False)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.proj(x)
        x= self.norm(x)
        return x


class Fusion(nn.Module):
    def __init__(self, dim, bias=False) -> None:
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 3, padding=1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

    def forward(self, in_1, in_2):
        if in_1.shape[-1] < in_2.shape[-1]:
            in_1 = F.interpolate(in_1, size=in_2.shape[-2:], mode='bilinear', align_corners=True)
        elif in_1.shape[-1] > in_2.shape[-1]:
            in_2 = F.interpolate(in_2, size=in_1.shape[-2:], mode='bilinear', align_corners=True)

        x = torch.cat((in_1, in_2), dim=1)
        x = self.fusion(x)
        return x


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(True)
    )