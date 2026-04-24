from .moganet import MogaNet


__all__ = ['moganet_xtiny', 'moganet_tiny', 'moganet_small',
           'moganet_base', 'moganet_large', 'moganet_xlarge']


class moganet_xtiny(MogaNet):
    def __init__(self, **kwargs):
        super(moganet_xtiny, self).__init__(
            embed_dims=[32, 64, 96, 192], depths=[3, 3, 10, 2], drop_path_rate=0.05, **kwargs)


class moganet_tiny(MogaNet):
    def __init__(self, **kwargs):
        super(moganet_tiny, self).__init__(
            embed_dims=[32, 64, 128, 256], depths=[3, 3, 12, 2], drop_path_rate=0.1, **kwargs)


class moganet_small(MogaNet):
    def __init__(self, **kwargs):
        super(moganet_small, self).__init__(
            embed_dims=[64, 128, 320, 512], depths=[2, 3, 12, 2], drop_path_rate=0.1, **kwargs)


class moganet_base(MogaNet):
    def __init__(self, **kwargs):
        super(moganet_base, self).__init__(
            embed_dims=[64, 160, 320, 512], depths=[4, 6, 22, 3], drop_path_rate=0.2, **kwargs)


class moganet_large(MogaNet):
    def __init__(self, **kwargs):
        super(moganet_large, self).__init__(
            embed_dims=[64, 160, 320, 640], depths=[4, 6, 44, 4], drop_path_rate=0.3, **kwargs)


class moganet_xlarge(MogaNet):
    def __init__(self, **kwargs):
        super(moganet_xlarge, self).__init__(
            embed_dims=[96, 192, 480, 960], depths=[6, 6, 44, 4], drop_path_rate=0.4, **kwargs)