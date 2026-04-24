import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_
from .utils import CSWinBlock, Merge_Block
from mmcv.runner import load_state_dict
from rsfm.module import CIM, build_ris_fusion, build_vg_fusion



class CSWin(nn.Module):
    '''
    from CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows, CVPR 2022
    '''
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 embed_dim=64,
                 depth=[1, 2, 21, 1],
                 split_size=[1, 2, 7, 7],
                 num_heads=[2, 4, 8, 16],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 with_cp=False,
                 l_dim=768,
                 vlf_ris=None,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.,
                 vlf_vg=None):
        super().__init__()
        self.with_cp = with_cp
        num_features = [int(embed_dim * 2 ** i) for i in range(len(depth))]
        
        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg
        
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, num_features[0], 7, 4, 2),
            Rearrange('b c h w -> b (h w) c', h=img_size // 4, w=img_size // 4),
            nn.LayerNorm(num_features[0]))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule

        self.norm1 = nn.LayerNorm(num_features[0])
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=num_features[0], num_heads=num_heads[0], patches_resolution=224 // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.merge1 = Merge_Block(num_features[0], num_features[1])

        self.norm2 = nn.LayerNorm(num_features[1])
        self.stage2 = nn.ModuleList(
            [CSWinBlock(
                dim=num_features[1], num_heads=num_heads[1], patches_resolution=224 // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depth[:1]) + i], norm_layer=norm_layer)
                for i in range(depth[1])])
        self.merge2 = Merge_Block(num_features[1], num_features[2])

        self.norm3 = nn.LayerNorm(num_features[2])
        self.stage3 = nn.ModuleList(
            [CSWinBlock(
                dim=num_features[2], num_heads=num_heads[2], patches_resolution=224 // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depth[:2]) + i], norm_layer=norm_layer)
                for i in range(depth[2])])
        self.merge3 = Merge_Block(num_features[2], num_features[3])

        self.stage4 = nn.ModuleList(
            [CSWinBlock(
                dim=num_features[3], num_heads=num_heads[3], patches_resolution=224 // 32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True)
                for i in range(depth[-1])])
        self.norm4 = norm_layer(num_features[3])

        if self.vlf_ris or self.vlf_vg:
            for i, vis_dim in enumerate(num_features):
                if self.vlf_ris:
                    fusion = build_ris_fusion(vlf_ris, vis_dim, l_dim, num_heads_fusion[i], fusion_drop,
                                              size=img_size // (2 ** (i + 2)))
                else:
                    fusion = build_vg_fusion(vlf_vg, vis_dim, l_dim, size=img_size // (2 ** (i + 2)))
                setattr(self, f"fusion{i + 1}", fusion)

        if self.vlf_ris == 'RMSIN':
            self.cim = CIM(dim=sum(num_features), channels=num_features, height=img_size // 32, width=img_size // 32)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
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

    def save_out(self, x, norm, H, W):
        x = norm(x)
        B, N, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x, l=None, l_mask=None):
        x = self.stage1_conv_embed[0](x)  ### B, C, H, W
        B, C, H, W = x.size()
        x = x.reshape(B, C, -1).transpose(-1, -2).contiguous()
        x = self.stage1_conv_embed[2](x)

        outs = []
        for blk in self.stage1:
            blk.H = H
            blk.W = W
            if self.with_cp:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.vlf_ris or self.vlf_vg:
            fusion = getattr(self, f"fusion1")
            if self.vlf_ris == 'DMMI':
                x, x_residual, l = fusion(x, l, l_mask)
            else:
                x, x_residual = fusion(x, l, l_mask)
            outs.append(self.save_out(x_residual, self.norm1, H, W))
        else:
            outs.append(self.save_out(x, self.norm1, H, W))

        for idx, (pre, blocks, norm) in enumerate(zip([self.merge1, self.merge2, self.merge3],
                                                      [self.stage2, self.stage3, self.stage4],
                                                      [self.norm2, self.norm3, self.norm4])):

            x, H, W = pre(x, H, W)
            for blk in blocks:
                blk.H = H
                blk.W = W
                if self.with_cp:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)

            if self.vlf_ris or self.vlf_vg:
                fusion = getattr(self, f"fusion{idx + 2}")
                if self.vlf_ris == 'DMMI':
                    x, x_residual, l = fusion(x, l, l_mask)
                else:
                    x, x_residual = fusion(x, l, l_mask)
                outs.append(self.save_out(x_residual, norm, H, W))
            else:
                outs.append(self.save_out(x, norm, H, W))

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        if self.vlf_ris == 'DMMI':
            return l, tuple(outs)
        else:
            return tuple(outs)