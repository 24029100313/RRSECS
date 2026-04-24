import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from mmcv.runner import load_state_dict
from rsfm.module import build_ris_fusion, CIM, build_vg_fusion
from .utils import StackConvPatchEmbed, ConvPatchEmbed, MogaBlock, build_norm_layer



class MogaNet(nn.Module):
    '''
    MogaNet: Multi-order Gated Aggregation Network, ICLR 2024
    '''
    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 embed_dims=[64, 160, 320, 512],
                 depths=[4, 6, 22, 3],
                 ffn_ratios=[8, 8, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 init_value=1e-5,
                 patch_sizes=[3, 3, 3, 3],
                 stem_norm_type='BN',
                 conv_norm_type='BN',
                 patchembed_types=['ConvEmbed', 'Conv', 'Conv', 'Conv'],
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_final_dilation=True,
                 attn_force_fp32=False,
                 l_dim=768,
                 vlf_ris=None,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.,
                 vlf_vg=None,
                 **kwargs):
        super().__init__()

        self.embed_dims = embed_dims
        self.depths = depths
        self.ffn_ratios = ffn_ratios

        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg

        self.num_stages = len(self.depths)
        self.attn_force_fp32 = attn_force_fp32
        self.use_layer_norm = stem_norm_type == 'LN'
        assert len(patchembed_types) == self.num_stages

        total_depth = sum(self.depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)] # stochastic depth decay rule

        cur_block_idx = 0
        for i, depth in enumerate(self.depths):
            if i == 0 and patchembed_types[i] == "ConvEmbed":
                assert patch_sizes[i] <= 3
                patch_embed = StackConvPatchEmbed(
                    in_channels=in_channels,
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    act_type='GELU',
                    norm_type=conv_norm_type,
                )
            else:
                patch_embed = ConvPatchEmbed(
                    in_channels=in_channels if i == 0 else self.embed_dims[i - 1],
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    norm_type=conv_norm_type)

            if i == self.num_stages - 1 and not attn_final_dilation:
                attn_dw_dilation = [1, 2, 1]
            blocks = nn.ModuleList([
                MogaBlock(
                    embed_dims=self.embed_dims[i],
                    ffn_ratio=self.ffn_ratios[i],
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[cur_block_idx + j],
                    norm_type=conv_norm_type,
                    init_value=init_value,
                    attn_dw_dilation=attn_dw_dilation,
                    attn_channel_split=attn_channel_split,
                    attn_act_type=attn_act_type,
                    attn_force_fp32=attn_force_fp32,
                ) for j in range(depth)
            ])
            cur_block_idx += depth

            norm = build_norm_layer(stem_norm_type, self.embed_dims[i])

            self.add_module(f'patch_embed{i + 1}', patch_embed)
            self.add_module(f'blocks{i + 1}', blocks)
            self.add_module(f'norm{i + 1}', norm)

            if self.vlf_ris:
                fusion = build_ris_fusion(vlf_ris, embed_dims[i], l_dim, num_heads_fusion[i], fusion_drop,
                                          size=img_size // (2 ** (i + 2)))
                self.add_module(f"fusion{i + 1}", fusion)

            if self.vlf_vg:
                fusion = build_vg_fusion(vlf_vg, embed_dims[i], l_dim, size=img_size // (2 ** (i + 2)))
                self.add_module(f"fusion{i + 1}", fusion)

        if self.vlf_ris == 'RMSIN':
            self.cim = CIM(dim=sum(embed_dims), channels=embed_dims, height=img_size // 32, width=img_size // 32)


    def init_weights(self, pretrained=None):
        def _init_weights(m):
            """ Init for timm image classification """
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

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
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            blocks = getattr(self, f'blocks{i + 1}')
            norm = getattr(self, f'norm{i + 1}')

            x, hw_shape = patch_embed(x)
            for block in blocks:
                x = block(x)

            if self.vlf_ris or self.vlf_vg:
                x = x.flatten(2).transpose(1, 2)
                fusion = getattr(self, f"fusion{i}")

                if self.vlf_ris == 'DMMI':
                    x, x_residual, l = fusion(x, l, l_mask)
                else:
                    x, x_residual = fusion(x, l, l_mask)

                if self.use_layer_norm:
                    x = norm(x)
                    x = x.reshape(-1, *hw_shape,
                                  blocks.out_channels).permute(0, 3, 1, 2).contiguous()
                    x_residual = norm(x_residual)
                    x_residual = x_residual.reshape(-1, *hw_shape,
                                                    blocks.out_channels).permute(0, 3, 1, 2).contiguous()
                else:
                    x = x.reshape(-1, *hw_shape,
                                  blocks.out_channels).permute(0, 3, 1, 2).contiguous()
                    x = norm(x)
                    x_residual = x_residual.reshape(-1, *hw_shape,
                                                    blocks.out_channels).permute(0, 3, 1, 2).contiguous()
                    x_residual = norm(x_residual)

                outs.append(x_residual)

            else:
                if self.use_layer_norm:
                    x = x.flatten(2).transpose(1, 2)
                    x = norm(x)
                    x = x.reshape(-1, *hw_shape,
                                  blocks.out_channels).permute(0, 3, 1, 2).contiguous()
                else:
                    x = norm(x)

                outs.append(x)

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        if self.vlf_ris == 'DMMI':
            return l, tuple(outs)
        else:
            return tuple(outs)