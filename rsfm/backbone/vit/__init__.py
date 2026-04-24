from .vit import VisionTransformer


__all__ = ['vit_base', 'vit_large']


class vit_tiny(VisionTransformer):
    def __init__(self, **kwargs):
        super(vit_tiny, self).__init__(
            patch_size=16, embed_dim=192, num_layers=12, num_heads=3, mlp_ratio=4, qkv_bias=True, drop_rate=0.0,
            attn_drop_rate=0.0, drop_path_rate=0.1, with_cls_token=True, norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'), norm_eval=False, interpolate_mode='bicubic', with_cp=False, **kwargs)


class vit_small(VisionTransformer):
    def __init__(self, **kwargs):
        super(vit_small, self).__init__(
            patch_size=16, embed_dim=384, num_layers=12, num_heads=6, mlp_ratio=4, qkv_bias=True, drop_rate=0.0,
            attn_drop_rate=0.0, drop_path_rate=0.1, with_cls_token=True, norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'), norm_eval=False, interpolate_mode='bicubic', with_cp=False, **kwargs)


class vit_base(VisionTransformer):
    def __init__(self, **kwargs):
        super(vit_base, self).__init__(
            patch_size=16, embed_dim=768, num_layers=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_rate=0.0,
            attn_drop_rate=0.0, drop_path_rate=0.1, with_cls_token=True, norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'), norm_eval=False, interpolate_mode='bicubic', with_cp=False, **kwargs)


class vit_large(VisionTransformer):
    def __init__(self, **kwargs):
        super(vit_large, self).__init__(
            patch_size=16, embed_dim=1024, num_layers=24, num_heads=16, mlp_ratio=4, qkv_bias=True, drop_rate=0.0,
            attn_drop_rate=0.0, drop_path_rate=0.2, with_cls_token=True, norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'), norm_eval=False, interpolate_mode='bicubic',
            out_indices=(7, 11, 15, 23), with_cp=True, **kwargs)