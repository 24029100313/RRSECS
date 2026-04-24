from torch import nn
from .utils import LayerNorm, PWAM
import torch.nn.functional as F
import math, torch



class MagNet_fusion(nn.Module):
    """
    from 'Mask Grounding for Referring Image Segmentation, CVPR 2024'
    """
    def __init__(self, dim, l_dim, num_heads_fusion=1, fusion_drop=0., pool_sizes=[1, 2, 3, 6]):
        super().__init__()

        self.fusion = PWAM(dim,  # both the visual input and for combining, num of channels
                           dim,  # v_in
                           l_dim,  # l_in
                           dim,  # key
                           dim,  # value
                           num_heads=num_heads_fusion,
                           dropout=fusion_drop)

        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.GELU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )

        self.pool_sizes = pool_sizes
        reduction_dim = dim // 4

        self.pyramids = nn.ModuleList()
        self.fusions = nn.ModuleList()

        for p in pool_sizes:
            self.pyramids.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(p),
                    nn.Conv2d(dim, dim * 4, kernel_size=1, bias=False),
                    LayerNorm(dim * 4),
                    nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False),
                    nn.GELU(),
                    nn.Conv2d(dim, reduction_dim, kernel_size=1, bias=False))
            )

            self.fusions.append(
                PWAM(reduction_dim,
                     reduction_dim,
                     l_dim,
                     reduction_dim,
                     reduction_dim,
                     num_heads=num_heads_fusion,
                     dropout=fusion_drop)
            )

            self.reduction_dim = reduction_dim

        self.mixer = nn.Sequential(nn.Linear(dim * 2, dim),
                                   nn.LayerNorm(dim),
                                   nn.Linear(dim, dim),
                                   nn.GELU())


    def forward(self, x, l, l_mask):
        out = []

        B, HW, C = x.shape
        H = W = int(math.sqrt(HW))

        x_reshape = x.permute(0, 2, 1).view(B, C, H, W)
        x_size = x_reshape.size()
        for i, p in enumerate(self.pool_sizes):
            px = self.pyramids[i](x_reshape)
            px = px.flatten(2).permute(0, 2, 1)
            px_residual = self.fusions[i](px, l, l_mask)
            px_residual = px_residual.permute(0, 2, 1).view(x.shape[0], self.reduction_dim, p, p)
            out.append(
                F.interpolate(px_residual, x_size[2:], mode='bilinear', align_corners=True).flatten(2).permute(0, 2, 1))

        # PWAM fusion
        x_residual = self.fusion(x, l, l_mask)

        out.append(x_residual)

        # apply a gate on the residual
        x = x + (self.res_gate(x_residual) * x_residual)

        x_residual = self.mixer(torch.cat(out, dim=2))

        return x, x_residual