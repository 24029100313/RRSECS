from torch import nn
from .utils import PWAM



class LAVT_fusion(nn.Module):
    """
    from 'LAVT: Language-Aware Vision Transformer for Referring Image Segmentation, CVPR 2022'
    """
    def __init__(self, dim, l_dim, num_heads_fusion=1, fusion_drop=0.):
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
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )

    def forward(self, x, l, l_mask):
        # PWAM fusion
        x_residual = self.fusion(x, l, l_mask)
        # apply a gate on the residual
        x = x + (self.res_gate(x_residual) * x_residual)

        return x, x_residual