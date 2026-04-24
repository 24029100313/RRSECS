from .focalnet import FocalNet


__all__ = ['focalnet_tiny', 'focalnet_small', 'focalnet_base', 'focalnet_large', 'focalnet_xlarge']


class focalnet_tiny(FocalNet):
    def __init__(self, **kwargs):
        super(focalnet_tiny, self).__init__(
            embed_dim=96, depths=[2, 2, 6, 2], drop_path_rate=0.3, patch_norm=True, with_cp=False,
            focal_windows=[9, 9, 9, 9], focal_levels=[3, 3, 3, 3], **kwargs)


class focalnet_small(FocalNet):
    def __init__(self, **kwargs):
        super(focalnet_small, self).__init__(
            embed_dim=96, depths=[2, 2, 18, 2], drop_path_rate=0.3, patch_norm=True, with_cp=False,
            focal_windows=[9, 9, 9, 9], focal_levels=[3, 3, 3, 3], **kwargs)


class focalnet_base(FocalNet):
    def __init__(self, **kwargs):
        super(focalnet_base, self).__init__(
            embed_dim=128, depths=[2, 2, 18, 2], drop_path_rate=0.3, patch_norm=True, with_cp=False,
            focal_windows=[9, 9, 9, 9], focal_levels=[3, 3, 3, 3], **kwargs)


class focalnet_large(FocalNet):
    def __init__(self, **kwargs):
        super(focalnet_large, self).__init__(
            embed_dim=192, depths=[2, 2, 18, 2], drop_path_rate=0.3, patch_norm=True, with_cp=False,
            focal_windows=[3, 3, 3, 3], focal_levels=[4, 4, 4, 4], patch_size=7, use_conv_embed=True,
            use_postln=True, use_postln_in_modulation=False, use_layerscale=True, normalize_modulator=True, **kwargs)


class focalnet_xlarge(FocalNet):
    def __init__(self, **kwargs):
        super(focalnet_xlarge, self).__init__(
            embed_dim=256, depths=[2, 2, 18, 2], drop_path_rate=0.3, patch_norm=True, with_cp=True,
            focal_windows=[3, 3, 3, 3], focal_levels=[4, 4, 4, 4], patch_size=7, use_conv_embed=True,
            use_postln=True, use_postln_in_modulation=False, use_layerscale=True, normalize_modulator=True, **kwargs)