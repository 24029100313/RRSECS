from .mambaout import MambaOut


__all__ = ['mambaout_femto', 'mambaout_kobe', 'mambaout_tiny', 'mambaout_small', 'mambaout_base']


class mambaout_femto(MambaOut):
    def __init__(self, **kwargs):
        super(mambaout_femto, self).__init__(
            depths=[3, 3, 9, 3], embed_dims=[48, 96, 192, 288], drop_path_rate=0.05, **kwargs)


class mambaout_kobe(MambaOut):
    def __init__(self, **kwargs):
        super(mambaout_kobe, self).__init__(
            depths=[3, 3, 15, 3], embed_dims=[48, 96, 192, 288], drop_path_rate=0.05, **kwargs)


class mambaout_tiny(MambaOut):
    def __init__(self, **kwargs):
        super(mambaout_tiny, self).__init__(
            depths=[3, 3, 9, 3], embed_dims=[96, 192, 384, 576], drop_path_rate=0.1, **kwargs)


class mambaout_small(MambaOut):
    def __init__(self, **kwargs):
        super(mambaout_small, self).__init__(
            depths=[3, 4, 27, 3], embed_dims=[96, 192, 384, 576], drop_path_rate=0.2, **kwargs)


class mambaout_base(MambaOut):
    def __init__(self, **kwargs):
        super(mambaout_base, self).__init__(
            depths=[3, 4, 27, 3], embed_dims=[128, 256, 512, 768], drop_path_rate=0.3, **kwargs)