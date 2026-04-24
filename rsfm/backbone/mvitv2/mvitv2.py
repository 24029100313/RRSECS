import math
from functools import partial
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from mmcv.runner import load_state_dict
import torch.utils.checkpoint as checkpoint
from rsfm.module import build_ris_fusion, CIM, build_vg_fusion
from .utils import MultiScaleBlock, round_width, PatchEmbed, prepare_mvit_configs



class MViTv2(nn.Module):
    """
    MViTv2: Improved Multiscale Vision Transformers for Classification and Detection, CVPR 2022
    """
    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 embed_dim=96,
                 num_heads=1,
                 depth=10,
                 use_abs_pos=False,
                 drop_path_rate=0.1,
                 mlp_ratio=4.0,
                 downscale_indices=[1, 3, 8],
                 pool_kernel=(3, 3),
                 adaptive_kv_stride=(4, 4),
                 qkv_bias=True,
                 rel_pos_spatial=True,
                 rel_pos_zero_init=False,
                 residual_pooling=True,
                 dim_mul_in_att=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 with_cp=False,
                 l_dim=768,
                 vlf_ris=None,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.,
                 vlf_vg=None):
        super().__init__()

        self.use_abs_pos = use_abs_pos
        self.with_cp = with_cp

        self.out_indices = [x - 1 for x in downscale_indices + [depth]]

        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg
        embed_dims = [embed_dim * 2 ** i for i in range(len(self.out_indices))]
        
        self.patch_embed = PatchEmbed(dim_in=in_channels, dim_out=embed_dim, 
                                      kernel=7, stride=4, padding=3)

        patch_dims = (img_size // 4, img_size // 4)
        num_patches = math.prod(patch_dims)

        if self.use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = None

        # MViT backbone configs
        dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv = prepare_mvit_configs(depth, downscale_indices,
                                                                                       pool_kernel, adaptive_kv_stride)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        input_size = patch_dims
        self.blocks = nn.ModuleList()
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            if dim_mul_in_att:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                rel_pos_spatial=rel_pos_spatial,
                rel_pos_zero_init=rel_pos_zero_init,
                residual_pooling=residual_pooling,
                dim_mul_in_att=dim_mul_in_att,
            )

            self.blocks.append(attention_block)

            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride for size, stride in zip(input_size, stride_q[i])
                ]
            embed_dim = dim_out

        if self.vlf_ris:
            for i, out_indice in enumerate(self.out_indices):
                fusion = build_ris_fusion(vlf_ris, embed_dims[i], l_dim, num_heads_fusion[i], fusion_drop,
                                          size=img_size // (2 ** (i + 2)))
                setattr(self, f"fusion{out_indice}", fusion)

        if self.vlf_vg:
            for i, out_indice in enumerate(self.out_indices):
                fusion = build_vg_fusion(vlf_vg, embed_dims[i], l_dim, size=img_size // (2 ** (i + 2)))
                setattr(self, f"fusion{out_indice}", fusion)

        if self.vlf_ris == 'RMSIN':
            self.cim = CIM(dim=sum(embed_dims), channels=embed_dims, height=img_size // 32, width=img_size // 32)

        if self.use_abs_pos:
            trunc_normal_(self.pos_embed, std=0.02)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
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
        return {k for k, _ in self.named_parameters()
                if any(n in k for n in ["pos_embed", "rel_pos_h", "rel_pos_w", "cls_token"])}

    def forward(self, x, l=None, l_mask=None):
        x, bchw = self.patch_embed(x)
        H, W = bchw[-2], bchw[-1]

        B = x.shape[0]
        outs = []

        if self.use_abs_pos:
            x = x + self.pos_embed

        thw = [H, W]
        for i, blk in enumerate(self.blocks):
            if self.with_cp:
                x, thw = checkpoint.checkpoint(blk, x, thw) # B, N, C
            else:
                x, thw = blk(x, thw)

            if i in self.out_indices:
                out = x
                if self.vlf_ris or self.vlf_vg:
                    if self.vlf_ris == 'DMMI':
                        x, out, l = self.__getattr__(f"fusion{i}")(out, l, l_mask)
                    else:
                        x, out = self.__getattr__(f"fusion{i}")(out, l, l_mask)

                out = out.reshape(B, thw[0], thw[1], -1).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        if self.vlf_ris == 'DMMI':
            return l, tuple(outs)
        else:
            return tuple(outs)


