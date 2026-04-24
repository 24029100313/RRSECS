from torch import nn
from .utils import PWAM



class DMMI_fusion(nn.Module):
    """
    from 'Beyond One-to-One: Rethinking the Referring Image Segmentation'
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

        self.W_l = nn.Sequential(
            nn.Conv1d(l_dim, l_dim, 1, 1),  # the init function sets bias to 0 if bias is True
            nn.GELU()
        )

    def forward(self, x, l, l_mask):
        x_residual, l_residual = self.fusion(x, l, l_mask)
        # apply a gate on the residual
        x = x + (self.res_gate(x_residual) * x_residual)
        l = l + self.W_l(l_residual)

        return x, x_residual, l