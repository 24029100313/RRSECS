from torch import nn
from .utils import QueryDynamicAttention



class QRNet_fusion(nn.Module):
    """
    from 'Shifting More Attention to Visual Backbone: Query-modulated Refinement Networks for End-to-End Visual Grounding, CVPR 2022'
    """
    def __init__(self, dim, l_dim, use_spatial=True, use_channel=True):
        super().__init__()

        self.qdatt = QueryDynamicAttention(gate_channels=dim,
                                           mu_dim=l_dim,
                                           reduction_ratio=16,
                                           pool_types=['avg', 'max'],
                                           use_spatial=use_spatial,
                                           use_channel=use_channel)

    def forward(self, x, l, l_mask):
        '''
        x: B, H*W, C
        l: B, l_dim
        '''
        x = self.qdatt(x, l)

        return x, x