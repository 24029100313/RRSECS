import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from mmcv.runner import load_state_dict
from .ops_dcnv3 import modules as opsm
from rsfm.module import CIM
from .utils import StemLayer, InternImageBlock



class InternImage(nn.Module):
    """
    InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions, CVPR 2023
    """
    def __init__(self,
                 img_size=224,
                 core_op='DCNv3',
                 in_channels=3,
                 embed_dim=64,
                 depths=[3, 4, 18, 5],
                 groups=[3, 6, 12, 24],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 drop_path_type='linear',
                 act_layer='GELU',
                 norm_layer='LN',
                 layer_scale=None,
                 offset_scale=1.0,
                 post_norm=False,
                 with_cp=False,
                 l_dim=768,
                 vlf_ris=None,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.,
                 vlf_vg=None,
                 dw_kernel_size=None,  # for InternImage-H/G
                 level2_post_norm=False,  # for InternImage-H/G
                 level2_post_norm_block_ids=None,  # for InternImage-H/G
                 res_post_norm=False,  # for InternImage-H/G
                 center_feature_scale=False,  # for InternImage-H/G
                 out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.core_op = core_op
        self.num_levels = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_levels - 1))
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio
        self.out_indices = out_indices
        self.in_channels = in_channels
        self.level2_post_norm_block_ids = level2_post_norm_block_ids

        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg

        self.patch_embed = StemLayer(in_chans=in_channels,
                                     out_chans=embed_dim,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        if drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            post_norm_block_ids = level2_post_norm_block_ids if level2_post_norm and (
                    i == 2) else None  # for InternImage-H/G
            level = InternImageBlock(
                core_op=getattr(opsm, core_op),
                channels=int(embed_dim * 2 ** i),
                depth=depths[i],
                groups=groups[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                post_norm_block_ids=post_norm_block_ids,  # for InternImage-H/G
                res_post_norm=res_post_norm,  # for InternImage-H/G
                center_feature_scale=center_feature_scale,  # for InternImage-H/G
                l_dim=l_dim,
                vlf_ris=self.vlf_ris,
                num_heads_fusion=num_heads_fusion[i],
                fusion_drop=fusion_drop,
                vlf_vg=self.vlf_vg,
                size=img_size // (2 ** (i + 2))
            )
            self.levels.append(level)

        if self.vlf_ris == 'RMSIN':
            dims = [int(embed_dim * 2 ** i) for i in range(self.num_levels)]
            self.cim = CIM(dim=sum(dims), channels=dims, height=img_size // 32, width=img_size // 32)

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

    def forward(self, x, l=None, l_mask=None):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        outs = []
        for level_idx, level in enumerate(self.levels):
            if self.vlf_ris == 'DMMI':
                x_out, x, l = level(x, l, l_mask)
            else:
                x_out, x = level(x, l, l_mask)

            if level_idx in self.out_indices:
                outs.append(x_out.permute(0, 3, 1, 2).contiguous())

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        if self.vlf_ris == 'DMMI':
            return l, tuple(outs)
        else:
            return tuple(outs)