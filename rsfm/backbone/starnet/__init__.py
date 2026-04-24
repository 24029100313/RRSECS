from .starnet import StarNet


__all__ = ['starnet_s1', 'starnet_s2', 'starnet_s3', 'starnet_s4']


class starnet_s1(StarNet):
    def __init__(self, **kwargs):
        super(starnet_s1, self).__init__(base_dim=24, depths=[2, 2, 8, 3], drop_path_rate=0.1, **kwargs)


class starnet_s2(StarNet):
    def __init__(self, **kwargs):
        super(starnet_s2, self).__init__(base_dim=32, depths=[1, 2, 6, 2], drop_path_rate=0.2, **kwargs)


class starnet_s3(StarNet):
    def __init__(self, **kwargs):
        super(starnet_s3, self).__init__(base_dim=32, depths=[2, 2, 8, 4], drop_path_rate=0.3, **kwargs)


class starnet_s4(StarNet):
    def __init__(self, **kwargs):
        super(starnet_s4, self).__init__(base_dim=32, depths=[3, 3, 12, 5], drop_path_rate=0.4, **kwargs)