from .gfnet import GFNet


__all__ = ['gfnet_tiny', 'gfnet_small', 'gfnet_base']


class gfnet_tiny(GFNet):
    def __init__(self, **kwargs):
        super(gfnet_tiny, self).__init__(
            embed_dims=[64, 128, 256, 512], depths=[3, 3, 10, 3], drop_path_rate=0.1, **kwargs)


class gfnet_small(GFNet):
    def __init__(self, **kwargs):
        super(gfnet_small, self).__init__(
            embed_dims=[96, 192, 384, 768], depths=[3, 3, 10, 3],
            drop_path_rate=0.2, in_channels=1e-5, **kwargs)


class gfnet_base(GFNet):
    def __init__(self, **kwargs):
        super(gfnet_base, self).__init__(
            embed_dims=[96, 192, 384, 768], depths=[3, 3, 27, 3],
            drop_path_rate=0.4, in_channels=1e-6, **kwargs)