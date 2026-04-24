from torch import nn
from rsfm.module.neck.utils import CBR



class Reshape(nn.Module):
    def forward(self, l):
        return l.view(l.shape[0], -1)


class CLM(nn.Module):
    '''Channel-wise Language-enhanced Module'''
    def __init__(self, dim, l_dim, reduction, dropout=0.):
        super(CLM, self).__init__()
        self.vis_proj = CBR(dim, dim, 1)

        self.lang_proj = nn.Sequential(nn.Linear(l_dim, l_dim),
                                       nn.GELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(l_dim, dim),
                                       nn.GELU())

        self.lang_channel_gate = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                               Reshape(),
                                               nn.Linear(dim, dim // reduction),
                                               nn.GELU(),
                                               nn.Linear(dim // reduction, dim),
                                               nn.Sigmoid())


        self.mix_proj = CBR(dim, dim, 1)

    def forward(self, x, l):
        x = self.vis_proj(x)
        l = self.lang_proj(l.permute(0, 2, 1))
        lcw = self.lang_channel_gate(l.permute(0, 2, 1)).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = self.mix_proj(x * lcw)

        return out