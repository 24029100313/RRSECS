from functools import partial
import torch.utils.checkpoint as checkpoint
from .utils import *
from mmcv.runner import load_state_dict


class Fast_iTPN(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=512,
                 depth_stage1=3,
                 depth_stage2=3,
                 depth=24,
                 num_heads=8,
                 bridge_mlp_ratio=3.,
                 mlp_ratio=3.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.0,
                 init_values=0.1,
                 attn_head_dim=None,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 patch_norm=False,
                 with_cp=False,
                 postnorm=False,
                 deepnorm=False,
                 subln=True,
                 swiglu=False,
                 naiveswiglu=True,
                 **kwargs):
        super().__init__()
        self.img_size = img_size
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp
        self.num_main_blocks = depth
        self.depth_stage1 = depth_stage1
        self.depth_stage2 = depth_stage2
        self.depth = depth
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim

        mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim}
        # split image into non-overlapping patches
        self.patch_embed = ConvPatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=mlvl_dims['4'],
            stop_grad_conv1=False, norm_layer=norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.subln = subln
        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu

        self.build_blocks(
            depths=[depth_stage1, depth_stage2, depth],
            dims=mlvl_dims,
            num_heads=num_heads,
            bridge_mlp_ratio=bridge_mlp_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            window_size=None,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            attn_head_dim=attn_head_dim,
            postnorm=postnorm,
            deepnorm=deepnorm,
            subln=subln,
            swiglu=swiglu,
            naiveswiglu=naiveswiglu,
        )

        ######## FPN stage #######
        fpn_dim = 256
        self.align_dim_16tofpn = nn.Linear(embed_dim, fpn_dim)
        self.fpn_modules = nn.ModuleList()
        self.fpn_modules.append(
            Block(dim=fpn_dim, num_heads=0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer))

        self.align_dim_16to8 = nn.Linear(mlvl_dims['8'], fpn_dim, bias=False)
        self.split_16to8 = PatchSplit(mlvl_dims['16'], fpn_dim, norm_layer)
        self.block_16to8 = nn.Sequential(
            *[Block(dim=fpn_dim, num_heads=0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer) for _ in range(1)])
        self.fpn_modules.append(
            Block(dim=fpn_dim, num_heads=0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer))

        self.align_dim_8to4 = nn.Linear(mlvl_dims['4'], fpn_dim, bias=False)
        self.split_8to4 = PatchSplit(fpn_dim, fpn_dim, norm_layer)
        self.block_8to4 = nn.Sequential(
            *[Block(dim=fpn_dim, num_heads=0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer) for _ in range(1)])
        self.fpn_modules.append(
            Block(dim=fpn_dim, num_heads=0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer))

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

    def build_blocks(self,
                     depths=[3, 3, 24],
                     dims={'4': 128 // 4, '8': 256, '16': 512},
                     num_heads=8,
                     bridge_mlp_ratio=3.,
                     mlp_ratio=4.0,
                     qkv_bias=True,
                     qk_scale=None,
                     window_size=None,
                     drop=0.,
                     attn_drop=0.,
                     drop_path_rate=0.,
                     norm_layer=nn.LayerNorm,
                     init_values=0.,
                     attn_head_dim=None,
                     postnorm=False,
                     deepnorm=False,
                     subln=False,
                     swiglu=False,
                     naiveswiglu=False,
                     ):
        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, depths[0] + depths[1] + depths[2]))

        self.blocks = nn.ModuleList()
        ######### stage 1 ########
        self.blocks.extend([
            ConvMlpBlock(
                dim=dims['4'],
                mlp_ratio=bridge_mlp_ratio,
                drop_path=next(dpr),
                norm_layer=norm_layer,
                init_values=0.,
                depth=depths[-1],
                postnorm=postnorm,
                deepnorm=deepnorm,
                subln=subln,
                swiglu=False,
                naiveswiglu=False,
            ) for _ in range(depths[0])
        ])
        self.blocks.append(ConvPatchMerge(dims['4'], norm_layer))

        ######### stage 2 ########
        self.blocks.extend([
            ConvMlpBlock(
                dim=dims['8'],
                mlp_ratio=bridge_mlp_ratio,
                drop_path=next(dpr),
                norm_layer=norm_layer,
                init_values=0.,
                depth=depths[-1],
                postnorm=postnorm,
                deepnorm=deepnorm,
                subln=subln,
                swiglu=False,
                naiveswiglu=False,
            ) for _ in range(depths[1])
        ])
        self.blocks.append(ConvPatchMerge(dims['8'], norm_layer))

        ######### stage 3 ########
        self.blocks.extend([
            Block(
                dim=dims['16'],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=next(dpr),
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=window_size,
                attn_head_dim=attn_head_dim,
                depth=depths[-1],
                postnorm=postnorm,
                deepnorm=deepnorm,
                subln=subln,
                swiglu=swiglu,
                naiveswiglu=naiveswiglu,
            ) for _ in range(depths[2])
        ])

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
                state_dict = checkpoint['module']
            load_state_dict(self, state_dict, False)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        for blk in self.blocks[:-self.num_main_blocks]:
            x = checkpoint.checkpoint(blk, x) if self.with_cp else blk(x)

        x = x.flatten(2).transpose(1, 2)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        for blk in self.blocks[-self.num_main_blocks:]:
            x = checkpoint.checkpoint(blk, x) if self.with_cp else blk(x)
