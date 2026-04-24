from torch import nn
from .utils import MCM, CrossAttentionBlock



class MSCAB(nn.Module):
    '''
    Multi-scale Cross-attention Block
    '''
    def __init__(self,
                 img_size=512,
                 in_channels=[96, 192, 384, 768],
                 out_channels=256,
                 l_dim=768,
                 num_heads=8,
                 dropout=0.):
        super(MSCAB, self).__init__()

        self.vis_proj_layers = nn.ModuleList([
            MCM(v_dim, out_channels) for v_dim in in_channels
        ])

        self.lang_proj_layer = nn.ModuleList([
            nn.Sequential(nn.Linear(l_dim, out_channels),
                          nn.GELU(),
                          nn.Dropout(dropout)) for _ in in_channels
        ])

        self.fusion_layers = nn.ModuleList([
            CrossAttentionBlock(out_channels, out_channels, num_heads, dropout,
                                window=img_size // (2 ** (i + 2))) for i in range(len(in_channels))
        ])

    def forward(self, inputs, l, l_mask=None):
        outs = []

        for idx, x in enumerate(inputs):
            xp = self.vis_proj_layers[idx](x)
            lp = self.lang_proj_layer[idx](l.permute(0, 2, 1))
            B, C, H, W = xp.shape
            xp = xp.flatten(2).transpose(1, 2)
            mm = self.fusion_layers[idx](xp, lp)
            out = mm.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(out)

        return outs
