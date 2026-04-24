from .mlla import MLLA


__all__ = ['mlla_tiny', 'mlla_small', 'mlla_base']



class mlla_tiny(MLLA):
    def __init__(self, **kwargs):
        super(mlla_tiny, self).__init__(
            embed_dim=64, depths=[2, 4, 8, 4], num_heads=[2, 4, 8, 16], drop_path_rate=0.1, **kwargs)


class mlla_small(MLLA):
    def __init__(self, **kwargs):
        super(mlla_small, self).__init__(
            embed_dim=64, depths=[3, 6, 21, 6], num_heads=[2, 4, 8, 16], drop_path_rate=0.2, **kwargs)


class mlla_base(MLLA):
    def __init__(self, **kwargs):
        super(mlla_base, self).__init__(
            embed_dim=96, depths=[3, 6, 21, 6], num_heads=[3, 6, 12, 24], drop_path_rate=0.3, **kwargs)