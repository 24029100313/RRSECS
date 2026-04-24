import torch, math
import torch.nn as nn
from functools import partial
from timm.layers import trunc_normal_
from mmcv.runner import load_state_dict
from .utils import PatchEmbed, DownLayer, BlockLayerScale
from rsfm.module import build_ris_fusion, CIM, build_vg_fusion



class GFNet(nn.Module):
    '''
    Global Filter Networks for Image Classification, NIPS 2021
    '''
    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 patch_size=4,
                 embed_dims=[64, 128, 256, 512],
                 depths=[3 ,3 ,10 ,3],
                 mlp_ratio=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 init_values=0.001,
                 l_dim=768,
                 vlf_ris=None,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.,
                 vlf_vg=None):
        super().__init__()
        self.depths = depths

        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg

        self.patch_embed = nn.ModuleList()
        patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dims[0])
        num_patches = patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]))

        self.patch_embed.append(patch_embed)

        sizes = [img_size // (2 ** (i + 2)) for i in range(len(self.depths))]

        for i in range(3):
            patch_embed = DownLayer(sizes[i], embed_dims[i], embed_dims[i + 1])
            self.patch_embed.append(patch_embed)

        self.blocks = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(len(self.depths)):
            h = sizes[i]
            w = h // 2 + 1

            blk = nn.Sequential(*[
                BlockLayerScale(
                    dim=embed_dims[i],
                    mlp_ratio=mlp_ratio[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    norm_layer=norm_layer,
                    h=h, w=w,
                    init_values=init_values)
                for j in range(depths[i])
            ])
            self.blocks.append(blk)

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"norm{i}", norm)

            if self.vlf_ris:
                fusion = build_ris_fusion(vlf_ris, embed_dims[i], l_dim, num_heads_fusion[i], fusion_drop,
                                          size=img_size // (2 ** (i + 2)))
                setattr(self, f"fusion{i}", fusion)

            if self.vlf_vg:
                fusion = build_vg_fusion(vlf_vg, embed_dims[i], l_dim, size=img_size // (2 ** (i + 2)))
                setattr(self, f"fusion{i}", fusion)

        if self.vlf_ris == 'RMSIN':
            self.cim = CIM(dim=sum(embed_dims), channels=embed_dims, height=img_size // 32, width=img_size // 32)

        trunc_normal_(self.pos_embed, std=.02)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            checkpoint = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            load_state_dict(self, state_dict, strict=False)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, l=None, l_mask=None):
        outs = []

        for i in range(len(self.depths)):
            norm = getattr(self, f"norm{i}")

            x = self.patch_embed[i](x)
            if i == 0:
                x = x + self.pos_embed
            x = self.blocks[i](x)

            B, N, C = x.shape
            H = W = int(math.sqrt(N))

            if self.vlf_ris or self.vlf_vg:
                fusion = getattr(self, f"fusion{i + 1}")

                if self.vlf_ris == 'DMMI':
                    x, x_residual, l = fusion(x, l, l_mask)
                else:
                    x, x_residual = fusion(x, l, l_mask)

                x_residual = norm(x_residual)
                x_residual = x_residual.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                outs.append(x_residual)
            else:
                out = norm(x)
                out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        if self.vlf_ris == 'DMMI':
            return l, tuple(outs)
        else:
            return tuple(outs)
