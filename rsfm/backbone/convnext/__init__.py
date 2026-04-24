from .convnext import ConvNeXt


__all__ = ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge']


class convnext_tiny(ConvNeXt):
    def __init__(self, **kwargs):
        super(convnext_tiny, self).__init__(
            embed_dim=96, depths=[3, 3, 9, 3], drop_path_rate=0.4, layer_scale_init_value=1.0, **kwargs)


class convnext_small(ConvNeXt):
    def __init__(self, **kwargs):
        super(convnext_small, self).__init__(
            embed_dim=96, depths=[3, 3, 27, 3], drop_path_rate=0.3, layer_scale_init_value=1.0, **kwargs)


class convnext_base(ConvNeXt):
    def __init__(self, **kwargs):
        super(convnext_base, self).__init__(
            embed_dim=128, depths=[3, 3, 27, 3], drop_path_rate=0.4, layer_scale_init_value=1.0, **kwargs)


class convnext_large(ConvNeXt):
    def __init__(self, **kwargs):
        super(convnext_large, self).__init__(
            embed_dim=192, depths=[3, 3, 27, 3], drop_path_rate=0.4, layer_scale_init_value=1.0, **kwargs)


class convnext_xlarge(ConvNeXt):
    def __init__(self, **kwargs):
        super(convnext_xlarge, self).__init__(
            embed_dim=256, depths=[3, 3, 27, 3], drop_path_rate=0.4, layer_scale_init_value=1.0, **kwargs)
