import torch.nn as nn
from rsfm.module.neck.mscab.utils import CrossAttentionModule


class LAGD(nn.Module):
    def __init__(self, v_dim, l_dim, n_heads=8, dropout=0.):
        super(LAGD, self).__init__()

        self.cmf = CrossAttentionModule(v_dim, l_dim, n_heads, dropout)

        self.gate = nn.Sequential(
            nn.Conv2d(v_dim, v_dim, 1),
            nn.BatchNorm2d(v_dim),
            nn.Sigmoid())

    def forward(self, x, l, l_mask):
        B, _, H, W = x.shape
        x = self.cmf(x.flatten(2).transpose(1, 2), l.permute(0, 2, 1))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        g = self.gate(x)
        ris_feat = g * x
        vg_feat = (1 - g) * x

        return ris_feat, vg_feat