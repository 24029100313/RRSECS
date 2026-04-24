from .inceptionnext import MetaNeXt


__all__ = ['inceptionnext_tiny', 'inceptionnext_small', 'inceptionnext_base']


class inceptionnext_tiny(MetaNeXt):
    def __init__(self, **kwargs):
        super(inceptionnext_tiny, self).__init__(
            depths=(3, 3, 9, 3), embed_dims=(96, 192, 384, 768), drop_path_rate=0.1, **kwargs)


class inceptionnext_small(MetaNeXt):
    def __init__(self, **kwargs):
        super(inceptionnext_small, self).__init__(
            depths=(3, 3, 27, 3), embed_dims=(96, 192, 384, 768), drop_path_rate=0.3, **kwargs)


class inceptionnext_base(MetaNeXt):
    def __init__(self, **kwargs):
        super(inceptionnext_base, self).__init__(
            depths=(3, 3, 27, 3), embed_dims=(128, 256, 512, 1024), drop_path_rate=0.4, **kwargs)