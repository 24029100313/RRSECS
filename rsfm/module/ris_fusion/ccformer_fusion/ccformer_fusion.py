from torch import nn
from .utils import FusionLayer
from rsfm.module.neck.mscab.utils import MCM



class CCFormer_fusion(nn.Module):
    def __init__(self, dim, l_dim, num_heads_fusion=1, fusion_drop=0.):
        super(CCFormer_fusion, self).__init__()

        self.vis_proj = MCM(dim, dim)

        self.fusion = FusionLayer(dim, l_dim, dim, num_heads_fusion, fusion_drop)

        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )

    def forward(self, x, l, l_mask):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x_proj = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_proj = self.vis_proj(x_proj).flatten(2).transpose(1, 2)

        x_residual = self.fusion(x_proj, l.permute(0, 2, 1), l_mask.permute(0, 2, 1))
        x = x + (self.res_gate(x_residual) * x_residual)

        return x, x_residual