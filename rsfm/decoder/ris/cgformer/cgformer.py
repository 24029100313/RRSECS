from torch import nn
from timm.models.layers import trunc_normal_
from .utils import CGAttention, DProjector, Fusion, LoadLayer


class CGFormerHead(nn.Module):
    '''
    from 'Contrastive Grouping with Transformer for Referring Image Segmentation'
    '''
    def __init__(self,
                 img_size=512,
                 in_channels=[128, 256, 512, 1024],
                 embedding_dim=512,
                 num_classes=2, **kwargs
                 ):
        super(CGFormerHead, self).__init__()

        token_dim = embedding_dim
        self.tokens = nn.Embedding(num_classes, token_dim)
        trunc_normal_(self.tokens.weight, std=0.02)

        dims = in_channels[::-1] # [1024, 512, 256, 128]
        pe_shapes = [img_size // 2 ** (i + 2) for i in range(2, -1, -1)]

        self.layers = []
        for pe_shape in pe_shapes:
            self.layers.append(LoadLayer(token_dim, drop=.1, bias=False, pe_shape=pe_shape))
        self.layers = nn.ModuleList(self.layers)

        self.cgattention1 = CGAttention(token_dim=token_dim,
                                        vis_dim=token_dim,
                                        hidden_dim=token_dim,
                                        drop=.1,
                                        bias=True)
        self.cgattention2 = CGAttention(token_dim=token_dim,
                                        vis_dim=token_dim,
                                        hidden_dim=token_dim,
                                        drop=.1,
                                        bias=True)

        self.fuses = []
        for i, dim in enumerate([dims[0], dims[2], dims[3]]):
            if i == 0:
                self.fuses.append(Fusion(dim, dims[1], token_dim, bias=True))
            else:
                self.fuses.append(Fusion(token_dim, dim, token_dim, bias=True))
        self.fuses = nn.ModuleList(self.fuses)

        self.proj = DProjector(text_dim=embedding_dim, in_dim=embedding_dim)

        self.conv_seg = nn.Conv2d(1, 2, 1)

    def forward(self, inputs, text_feat, text_mask):
        x_c1, x_c2, x_c3, x_c4 = inputs

        tokens = self.tokens.weight[None, ...].expand(x_c1.shape[0], -1, -1)

        maps = []
        v = x_c4
        for load, layer, fuse, v_ in zip(self.layers,
                                         [self.cgattention1, self.cgattention2, self.cgattention2],
                                         self.fuses,
                                         [x_c3, x_c2, x_c1]):
            v = fuse(v, v_)
            tokens, pe = load(tokens, text_feat, text_mask)
            tokens, hitmap = layer(tokens, v, pe=pe)
            maps.append(hitmap)

        out = self.proj(v, tokens[:, -1])

        out = self.conv_seg(out)

        return out
