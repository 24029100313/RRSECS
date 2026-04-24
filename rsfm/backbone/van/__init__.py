from .van import VAN


__all__ = ['van_b0', 'van_b1', 'van_b2', 'van_b3', 'van_b4', 'van_b5', 'van_b6']


class van_b0(VAN):
    def __init__(self, **kwargs):
        super(van_b0, self).__init__(
            embed_dim=32, depths=[3, 3, 5, 2], drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class van_b1(VAN):
    def __init__(self, **kwargs):
        super(van_b1, self).__init__(
            embed_dim=64, depths=[2, 2, 4, 2], drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class van_b2(VAN):
    def __init__(self, **kwargs):
        super(van_b2, self).__init__(
            embed_dim=64, depths=[3, 3, 12, 3], drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class van_b3(VAN):
    def __init__(self, **kwargs):
        super(van_b3, self).__init__(
            embed_dim=64, depths=[3, 5, 27, 3], drop_rate=0.0, drop_path_rate=0.3, **kwargs)


class van_b4(VAN):
    def __init__(self, **kwargs):
        super(van_b4, self).__init__(
            embed_dim=64, depths=[3, 6, 40, 3], drop_rate=0.0, drop_path_rate=0.4, **kwargs)


class van_b5(VAN):
    def __init__(self, **kwargs):
        super(van_b5, self).__init__(
            embed_dim=96, depths=[3, 3, 24, 3], drop_rate=0.0, drop_path_rate=0.4, **kwargs)


class van_b6(VAN):
    def __init__(self, **kwargs):
        super(van_b6, self).__init__(
            embed_dim=96, num_heads=[1, 2, 4, 8], depths=[6, 6, 90, 6], drop_rate=0.0, drop_path_rate=0.5, **kwargs)