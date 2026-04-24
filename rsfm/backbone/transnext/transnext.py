import math
import torch
import torch.nn as nn
from functools import partial
from mmcv.runner import load_state_dict
from timm.models.layers import to_2tuple, trunc_normal_
from rsfm.module import build_ris_fusion, CIM, build_vg_fusion
from .utils import get_relative_position_cpb, OverlapPatchEmbed, Block, get_seqlen_and_mask




class TransNeXt(nn.Module):
    '''
    TransNeXt: Robust Foveal Visual Perception for Vision Transformers, CVPR 2024
    '''
    def __init__(self,
                 img_size=224,
                 pretrain_size=224,
                 window_size=[3, 3, 3, None],
                 patch_size=4,
                 in_channels=3,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[8, 8, 4, 4],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 num_stages=4,
                 is_extrapolation=False,
                 l_dim=768,
                 vlf_ris=None,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.,
                 vlf_vg=None):
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages
        self.window_size = window_size
        self.sr_ratios = sr_ratios
        self.is_extrapolation = is_extrapolation
        self.pretrain_size = pretrain_size or img_size

        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if not self.is_extrapolation:
                relative_pos_index, relative_coords_table = get_relative_position_cpb(
                    query_size=to_2tuple(img_size // (2 ** (i + 2))),
                    key_size=to_2tuple(img_size // ((2 ** (i + 2)) * sr_ratios[i])),
                    pretrain_size=to_2tuple(pretrain_size // (2 ** (i + 2))))

                self.register_buffer(f"relative_pos_index{i + 1}", relative_pos_index, persistent=False)
                self.register_buffer(f"relative_coords_table{i + 1}", relative_coords_table, persistent=False)

            patch_embed = OverlapPatchEmbed(patch_size=patch_size * 2 - 1 if i == 0 else 3,
                                            stride=patch_size if i == 0 else 2,
                                            in_chans=in_channels if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], input_resolution=to_2tuple(img_size // (2 ** (i + 2))), window_size=window_size[i],
                num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], is_extrapolation=is_extrapolation)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

            if self.vlf_ris:
                fusion = build_ris_fusion(vlf_ris, embed_dims[i], l_dim, num_heads_fusion[i], fusion_drop,
                                          size=img_size // (2 ** (i + 2)))
                setattr(self, f"fusion{i + 1}", fusion)

            if self.vlf_vg:
                fusion = build_vg_fusion(vlf_vg, embed_dims[i], l_dim, size=img_size // (2 ** (i + 2)))
                setattr(self, f"fusion{i + 1}", fusion)

        if self.vlf_ris == 'RMSIN':
            self.cim = CIM(dim=sum(embed_dims), channels=embed_dims, height=img_size // 32, width=img_size // 32)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

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
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'query_embedding', 'relative_pos_bias_local', 'cpb', 'temperature'}

    def forward(self, x, l=None, l_mask=None):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            sr_ratio = self.sr_ratios[i]
            if self.is_extrapolation:
                relative_pos_index, relative_coords_table = get_relative_position_cpb(query_size=(H, W),
                                                                                      key_size=(
                                                                                          H // sr_ratio,
                                                                                          W // sr_ratio),
                                                                                      pretrain_size=to_2tuple(
                                                                                          self.pretrain_size // (
                                                                                                  2 ** (i + 2))),
                                                                                      device=x.device)
            else:
                relative_pos_index = getattr(self, f"relative_pos_index{i + 1}")
                relative_coords_table = getattr(self, f"relative_coords_table{i + 1}")

            with torch.no_grad():
                if i != (self.num_stages - 1):
                    local_seq_length, padding_mask = get_seqlen_and_mask((H, W), self.window_size[i], device=x.device)
                    seq_length_scale = torch.log(local_seq_length + (H // sr_ratio) * (W // sr_ratio))
                else:
                    seq_length_scale = torch.log(torch.as_tensor((H // sr_ratio) * (W // sr_ratio), device=x.device))
                    padding_mask = None
            for blk in block:
                x = blk(x, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask)

            if self.vlf_ris or self.vlf_vg:
                fusion = getattr(self, f"fusion{i + 1}")

                if self.vlf_ris == 'DMMI':
                    x, x_residual, l = fusion(x, l, l_mask)
                else:
                    x, x_residual = fusion(x, l, l_mask)

                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                x_residual = norm(x_residual)
                x_residual = x_residual.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                outs.append(x_residual)
            else:
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                outs.append(x)

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        if self.vlf_ris == 'DMMI':
            return l, tuple(outs)
        else:
            return tuple(outs)
