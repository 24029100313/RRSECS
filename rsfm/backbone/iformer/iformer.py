import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_state_dict
from rsfm.module import build_ris_fusion, CIM, build_vg_fusion
from .utils import PatchEmbed, FirstPatchEmbed, Block, trunc_normal_



class InceptionTransformer(nn.Module):
    '''
    Inception Transformer, NIPS 2022
    '''
    def __init__(self, 
                 img_size=224,
                 in_channels=3,
                 embed_dims=None,
                 depths=None,
                 num_heads=None, 
                 mlp_ratio=4., 
                 qkv_bias=True,
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 embed_layer=PatchEmbed, 
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 attention_heads=None,
                 use_layer_scale=False, 
                 layer_scale_init_value=1e-5,
                 l_dim=768,
                 vlf_ris=None,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.,
                 vlf_vg=None):
        super().__init__()

        st2_idx = sum(depths[:1])
        st3_idx = sum(depths[:2])
        st4_idx = sum(depths[:3])
        depth = sum(depths)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.patch_embed = FirstPatchEmbed(in_chans=in_channels, embed_dim=embed_dims[0])
        self.num_patches1 = num_patches = img_size // 4
        self.pos_embed1 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[0]))
        self.blocks1 = nn.Sequential(*[
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                attention_head=attention_heads[i], pool_size=2, )
            # use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, 
            # )
            for i in range(0, st2_idx)])
        self.norm1 = norm_layer(embed_dims[0])

        self.patch_embed2 = embed_layer(kernel_size=3, stride=2, padding=1, in_chans=embed_dims[0],
                                        embed_dim=embed_dims[1])
        self.num_patches2 = num_patches = num_patches // 2
        self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[1]))
        self.blocks2 = nn.Sequential(*[
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                attention_head=attention_heads[i], pool_size=2, )
            # use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value, channel_layer_scale=channel_layer_scale,
            # )
            for i in range(st2_idx, st3_idx)])
        self.norm2 = norm_layer(embed_dims[1])

        self.patch_embed3 = embed_layer(kernel_size=3, stride=2, padding=1, in_chans=embed_dims[1],
                                        embed_dim=embed_dims[2])
        self.num_patches3 = num_patches = num_patches // 2
        self.pos_embed3 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[2]))
        self.blocks3 = nn.Sequential(*[
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                attention_head=attention_heads[i], pool_size=1,
                use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value,
            )
            for i in range(st3_idx, st4_idx)])
        self.norm3 = norm_layer(embed_dims[2])

        self.patch_embed4 = embed_layer(kernel_size=3, stride=2, padding=1, in_chans=embed_dims[2],
                                        embed_dim=embed_dims[3])
        self.num_patches4 = num_patches = num_patches // 2
        self.pos_embed4 = nn.Parameter(torch.zeros(1, num_patches, num_patches, embed_dims[3]))
        self.blocks4 = nn.Sequential(*[
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                attention_head=attention_heads[i], pool_size=1,
                use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value,
            )
            for i in range(st4_idx, depth)])
        self.norm4 = norm_layer(embed_dims[3])

        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg

        if self.vlf_ris or self.vlf_vg:
            for i, vis_dim in enumerate(embed_dims):
                if self.vlf_ris:
                    fusion = build_ris_fusion(vlf_ris, vis_dim, l_dim, num_heads_fusion[i], fusion_drop,
                                              size=img_size // (2 ** (i + 2)))
                else:
                    fusion = build_vg_fusion(vlf_vg, vis_dim, l_dim, size=img_size // (2 ** (i + 2)))
                setattr(self, f"fusion{i + 1}", fusion)

        if self.vlf_ris == 'RMSIN':
            self.cim = CIM(dim=sum(embed_dims), channels=embed_dims, height=img_size // 32, width=img_size // 32)


    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        return {'pos_embed', 'cls_token', 'dist_token'}

    def _get_pos_embed(self, pos_embed, num_patches_def, H, W):
        if H * W == num_patches_def * num_patches_def:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").permute(0, 2, 3, 1)

    def save_out(self, x, norm, H, W):
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        x = norm(x)
        B, N, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x, l=None, l_mask=None):
        outs = []

        x = self.patch_embed(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed1, self.num_patches1, H, W)
        x = self.blocks1(x) # B, H, W, C
        if self.vlf_ris or self.vlf_vg:
            fusion = getattr(self, f"fusion1")
            x = x.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            if self.vlf_ris == 'DMMI':
                x, x_residual, l = fusion(x, l, l_mask)
            else:
                x, x_residual = fusion(x, l, l_mask)
            x = x.view(B, H, W, C)
            outs.append(self.save_out(x_residual, self.norm1, H, W))
        else:
            outs.append(self.save_out(x, self.norm1, H, W))

        x = x.permute(0, 3, 1, 2)
        x = self.patch_embed2(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed2, self.num_patches2, H, W)
        x = self.blocks2(x)
        if self.vlf_ris or self.vlf_vg:
            fusion = getattr(self, f"fusion2")
            x = x.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            if self.vlf_ris == 'DMMI':
                x, x_residual, l = fusion(x, l, l_mask)
            else:
                x, x_residual = fusion(x, l, l_mask)
            x = x.view(B, H, W, C)
            outs.append(self.save_out(x_residual, self.norm2, H, W))
        else:
            outs.append(self.save_out(x, self.norm2, H, W))

        x = x.permute(0, 3, 1, 2)
        x = self.patch_embed3(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed3, self.num_patches3, H, W)
        x = self.blocks3(x)
        if self.vlf_ris or self.vlf_vg:
            fusion = getattr(self, f"fusion3")
            x = x.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            if self.vlf_ris == 'DMMI':
                x, x_residual, l = fusion(x, l, l_mask)
            else:
                x, x_residual = fusion(x, l, l_mask)
            x = x.view(B, H, W, C)
            outs.append(self.save_out(x_residual, self.norm3, H, W))
        else:
            outs.append(self.save_out(x, self.norm3, H, W))

        x = x.permute(0, 3, 1, 2)
        x = self.patch_embed4(x)
        B, H, W, C = x.shape
        x = x + self._get_pos_embed(self.pos_embed4, self.num_patches4, H, W)
        x = self.blocks4(x)
        if self.vlf_ris or self.vlf_vg:
            fusion = getattr(self, f"fusion4")
            x = x.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            if self.vlf_ris == 'DMMI':
                x, x_residual, l = fusion(x, l, l_mask)
            else:
                x, x_residual = fusion(x, l, l_mask)
            x = x.view(B, H, W, C)
            outs.append(self.save_out(x_residual, self.norm4, H, W))
        else:
            outs.append(self.save_out(x, self.norm4, H, W))

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        if self.vlf_ris == 'DMMI':
            return l, tuple(outs)
        else:
            return tuple(outs)
