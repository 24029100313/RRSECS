from .internimage import InternImage


__all__ = ['internimage_tiny', 'internimage_small', 'internimage_base',
           'internimage_large', 'internimage_xlarge', 'internimage_huge']


class internimage_tiny(InternImage):
    def __init__(self, **kwargs):
        super(internimage_tiny, self).__init__(
            embed_dim=64, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32], mlp_ratio=4., drop_path_rate=0.2,
            norm_layer='LN', layer_scale=1.0, offset_scale=1.0, post_norm=False, with_cp=False, **kwargs)


class internimage_small(InternImage):
    def __init__(self, **kwargs):
        super(internimage_small, self).__init__(
            embed_dim=80, depths=[4, 4, 21, 4], groups=[5, 10, 20, 40], mlp_ratio=4., drop_path_rate=0.3,
            norm_layer='LN', layer_scale=1.0, offset_scale=1.0, post_norm=True, with_cp=False, **kwargs)


class internimage_base(InternImage):
    def __init__(self, **kwargs):
        super(internimage_base, self).__init__(
            embed_dim=112, depths=[4, 4, 21, 4], groups=[7, 14, 28, 56], mlp_ratio=4., drop_path_rate=0.4,
            norm_layer='LN', layer_scale=1.0, offset_scale=1.0, post_norm=True, with_cp=False, **kwargs)


class internimage_large(InternImage):
    def __init__(self, **kwargs):
        super(internimage_large, self).__init__(
            embed_dim=160, depths=[5, 5, 22, 5], groups=[10, 20, 40, 80], mlp_ratio=4., drop_path_rate=0.4,
            norm_layer='LN', layer_scale=1.0, offset_scale=2.0, post_norm=True, with_cp=False, **kwargs)


class internimage_xlarge(InternImage):
    def __init__(self, **kwargs):
        super(internimage_xlarge, self).__init__(
            embed_dim=192, depths=[5, 5, 24, 5], groups=[12, 24, 48, 96], mlp_ratio=4., drop_path_rate=0.4,
            norm_layer='LN', layer_scale=1.0, offset_scale=2.0, post_norm=True, with_cp=False, **kwargs)


class internimage_huge(InternImage):
    def __init__(self, **kwargs):
        super(internimage_huge, self).__init__(
            embed_dim=320, depths=[6, 6, 32, 6], groups=[10, 20, 40, 80], mlp_ratio=4., drop_path_rate=0.5,
            norm_layer='LN', layer_scale=None, offset_scale=1.0, post_norm=False, dw_kernel_size=5,
            res_post_norm=True, level2_post_norm=True, level2_post_norm_block_ids=[5, 11, 17, 23, 29],
            center_feature_scale=True, with_cp=True, **kwargs)