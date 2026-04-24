from .transnext import TransNeXt


__all__ = ['transnext_micro', 'transnext_tiny', 'transnext_small', 'transnext_base']


class transnext_micro(TransNeXt):
    def __init__(self, **kwargs):
        super(transnext_micro, self).__init__(embed_dims=[48, 96, 192, 384], num_heads=[2, 4, 8, 16],
                                              depths=[2, 2, 15, 2], drop_path_rate=0.3, **kwargs)


class transnext_tiny(TransNeXt):
    def __init__(self, **kwargs):
        super(transnext_tiny, self).__init__(embed_dims=[72, 144, 288, 576], num_heads=[3, 6, 12, 24],
                                             depths=[2, 2, 15, 2], drop_path_rate=0.4, **kwargs)


class transnext_small(TransNeXt):
    def __init__(self, **kwargs):
        super(transnext_small, self).__init__(embed_dims=[72, 144, 288, 576], num_heads=[3, 6, 12, 24],
                                              depths=[5, 5, 22, 5], drop_path_rate=0.6, **kwargs)


class transnext_base(TransNeXt):
    def __init__(self, **kwargs):
        super(transnext_base, self).__init__(embed_dims=[96, 192, 384, 768], num_heads=[4, 8, 16, 32],
                                             depths=[5, 5, 23, 5], drop_path_rate=0.7, **kwargs)
