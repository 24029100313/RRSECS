from .mvitv2 import MViTv2


__all__ = ['mvitv2_tiny', 'mvitv2_small', 'mvitv2_base', 'mvitv2_large', 'mvitv2_huge']


class mvitv2_tiny(MViTv2):
    def __init__(self, **kwargs):
        super(mvitv2_tiny, self).__init__(embed_dim=96, depth=10, downscale_indices=[1, 3, 8],
                                          drop_path_rate=0.2, **kwargs)


class mvitv2_small(MViTv2):
    def __init__(self, **kwargs):
        super(mvitv2_small, self).__init__(embed_dim=96, depth=16, downscale_indices=[1, 3, 14],
                                           drop_path_rate=0.3, **kwargs)


class mvitv2_base(MViTv2):
    def __init__(self, **kwargs):
        super(mvitv2_base, self).__init__(embed_dim=96, depth=24, downscale_indices=[2, 5, 21],
                                          drop_path_rate=0.4, **kwargs)


class mvitv2_large(MViTv2):
    def __init__(self, **kwargs):
        super(mvitv2_large, self).__init__(embed_dim=144, depth=48, num_heads=2, downscale_indices=[2, 8, 44],
                                           drop_path_rate=0.5, **kwargs)


class mvitv2_huge(MViTv2):
    def __init__(self, **kwargs):
        super(mvitv2_huge, self).__init__(embed_dim=192, depth=80, num_heads=3, downscale_indices=[4, 12, 72],
                                          drop_path_rate=0.6, **kwargs)