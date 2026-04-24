from rsfm.module.ris_fusion import LAVT_fusion
from torch import nn
from .utils import VariousReceptive




class RMSIN_fusion(LAVT_fusion):
    """
    from 'Rotated Multi-Scale Interaction Network for Referring Remote Sensing Image Segmentation, CVPR 2024'
    """
    def __init__(self, dim, l_dim, num_heads_fusion=1, fusion_drop=0.):
        super(RMSIN_fusion, self).__init__(dim=dim, l_dim=l_dim,
                                           num_heads_fusion=num_heads_fusion, fusion_drop=fusion_drop)

        # Various Receptive Branch
        self.visual_residual = VariousReceptive(dim)

        self.visual_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )

    def forward(self, x, l, l_mask):
        v_residual = self.visual_residual(x)
        # PWAM fusion
        x_residual = self.fusion(x, l, l_mask)
        # apply a gate on the residual
        x = x + (self.res_gate(x_residual) * x_residual) + (self.visual_gate(v_residual) * v_residual)

        return x, x_residual