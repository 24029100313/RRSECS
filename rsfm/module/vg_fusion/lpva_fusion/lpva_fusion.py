from torch import nn
from .utils import PA



class LPVA_fusion(nn.Module):
    """
    from 'Language-Guided Progressive Attention for Visual Grounding in Remote Sensing Images, TGRS 2024'
    """
    def __init__(self, dim, l_dim, size):
        '''
        size: visual feature size at each stage
        '''
        super().__init__()

        self.pa = PA(v_channels=dim,
                     l_channels=l_dim,
                     size=size)

    def forward(self, x, l, l_mask):
        '''
        x: B, H*W, C
        l: B, N, l_dim
        '''
        x = self.pa(x, l)

        return x, x