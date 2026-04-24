import math
import torch
import torch.nn as nn
from ..vit import VisionTransformer
from .pos_embed import get_2d_sincos_pos_embed
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import CheckpointLoader, load_state_dict
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init, trunc_normal_)


class ViT_cross_scalemae(VisionTransformer):
    def __init__(self, **kwargs):
        super(ViT_cross_scalemae, self).__init__(**kwargs)

        self.num_patches = (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            checkpoint = CheckpointLoader.load_checkpoint(pretrained, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    h, w = self.img_size
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size), self.interpolate_mode)

            load_state_dict(self, state_dict, strict=False)
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    def forward(self, x, l=None, l_mask=None):
        B = x.shape[0]
        x, hw_shape = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.drop_after_pos(x)

        if not self.with_cls_token:
            x = x[:, 1:]

        outs = []
        if self.vlf_ris:
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i == len(self.layers) - 1:
                    if self.final_norm:
                        x = self.norm1(x)
                if i in self.out_indices:
                    x, x_residual = self.__getattr__(f"fusion{i}")(x, l, l_mask)

                    if self.with_cls_token:
                        # Remove class token and reshape token for decoder head
                        out = x_residual[:, 1:]
                    else:
                        out = x_residual
                    B, _, C = out.shape
                    out = out.reshape(B, hw_shape[0], hw_shape[1],
                                      C).permute(0, 3, 1, 2).contiguous()
                    if self.output_cls_token:
                        out = [out, x_residual[:, 0]]
                    outs.append(out)
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i == len(self.layers) - 1:
                    if self.final_norm:
                        x = self.norm1(x)
                if i in self.out_indices:
                    if self.with_cls_token:
                        # Remove class token and reshape token for decoder head
                        out = x[:, 1:]
                    else:
                        out = x
                    B, _, C = out.shape
                    out = out.reshape(B, hw_shape[0], hw_shape[1],
                                      C).permute(0, 3, 1, 2).contiguous()
                    if self.output_cls_token:
                        out = [out, x[:, 0]]
                    outs.append(out)

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        return tuple(outs)


class vit_large_patch16_cross_scalemae_rsfm(ViT_cross_scalemae):
    def __init__(self, **kwargs):
        super(vit_large_patch16_cross_scalemae_rsfm, self).__init__(
            patch_size=16, embed_dim=1024, num_layers=24, num_heads=16, mlp_ratio=4, qkv_bias=True, drop_rate=0.0,
            attn_drop_rate=0.0, drop_path_rate=0.2, with_cls_token=True, norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'), norm_eval=False, interpolate_mode='bicubic',
            out_indices=(7, 11, 15, 23), with_cp=True, **kwargs)