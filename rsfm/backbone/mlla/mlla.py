import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from mmcv.runner import load_state_dict
from rsfm.module import CIM
from .utils import Stem, BasicLayer, PatchMerging



class MLLA(nn.Module):
    '''
    Demystify Mamba in Vision: A Linear Attention Perspective, NIPS 2024
    '''
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 out_indices=(0, 1, 2, 3),
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 with_cp=False,
                 l_dim=768,
                 vlf_ris=None,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.,
                 vlf_vg=None,
                 **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.out_indices = out_indices
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            stages_norm_layer = norm_layer(self.num_features[i])
            stages_norm_layer_name = f'norm{i}'
            self.add_module(stages_norm_layer_name, stages_norm_layer)
        self.mlp_ratio = mlp_ratio

        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg

        self.patch_embed = Stem(img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                with_cp=with_cp,
                l_dim=l_dim,
                vlf_ris=self.vlf_ris,
                num_heads_fusion=num_heads_fusion[i_layer],
                fusion_drop=fusion_drop,
                vlf_vg=self.vlf_vg,
                size=img_size // (2 ** (i_layer + 2))
            )
            self.layers.append(layer)

        if self.vlf_ris == 'RMSIN':
            self.cim = CIM(dim=sum(self.num_features), channels=self.num_features,
                           height=img_size//32, width=img_size//32)


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
            load_state_dict(self, state_dict, False)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x, l=None, l_mask=None):
        x, hw_shape = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        outs = []
        for i, layer in enumerate(self.layers):
            if self.vlf_ris == 'DMMI':
                x, hw_shape, out, l, out_hw_shape = layer(x, hw_shape, l, l_mask)
            else:
                x, hw_shape, out, out_hw_shape = layer(x, hw_shape, l, l_mask)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        if self.vlf_ris == 'DMMI':
            return l, tuple(outs)
        else:
            return tuple(outs)