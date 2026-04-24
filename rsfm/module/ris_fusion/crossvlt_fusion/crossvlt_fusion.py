from torch import nn
from .utils import FusionLayer


class CrossVLT_fusion(nn.Module):
    '''
    from 'Cross-aware Early Fusion with Stage-divided Vision and Language Transformer Encoders for Referring Image Segmentation'
    '''
    def __init__(self, dim, l_dim, num_heads_fusion=1, fusion_drop=0.):
        super(CrossVLT_fusion, self).__init__()

        self.fusion = FusionLayer(dim, l_dim, dim, num_heads_fusion, fusion_drop)

        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )

    def forward(self, x, l, l_mask):
        x_residual = self.fusion(x, l.permute(0, 2, 1), l_mask.permute(0, 2, 1))
        x = x + (self.res_gate(x_residual) * x_residual)

        return x, x_residual