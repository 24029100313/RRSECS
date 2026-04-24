from torch import nn
from rsfm.module.neck.mscab.utils import MCM, CrossAttentionBlock



class MCT_fusion(nn.Module):
    '''
    Multi-scale Cross-attention Transformer for Referring Remote Sensing Image Segmentation
    '''
    def __init__(self,
                 dim=96,
                 l_dim=768,
                 num_heads=8,
                 dropout=0.,
                 size=16):
        super(MCT_fusion, self).__init__()

        self.vis_proj = MCM(dim, dim)

        self.lang_proj = nn.Sequential(nn.Linear(l_dim, l_dim),
                                       nn.GELU(),
                                       nn.Dropout(dropout))

        self.fusion = CrossAttentionBlock(dim, l_dim, num_heads, dropout, size)

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
        l_proj = self.lang_proj(l.permute(0, 2, 1))

        x_residual = self.fusion(x_proj, l_proj, l_mask)
        x = x + (self.res_gate(x_residual) * x_residual)

        return x, x_residual
