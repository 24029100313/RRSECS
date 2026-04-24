from .unireplknet import UniRepLKNet


__all__ = ['unireplknet_tiny', 'unireplknet_small', 'unireplknet_base', 'unireplknet_large', 'unireplknet_xlarge']



class unireplknet_tiny(UniRepLKNet):
    def __init__(self, **kwargs):
        super(unireplknet_tiny, self).__init__(
            depths=[3, 3, 18, 3], embed_dim=80, drop_path_rate=0.2,
            kernel_sizes=None, with_cp=False, attempt_use_lk_impl=False, **kwargs)


class unireplknet_small(UniRepLKNet):
    def __init__(self, **kwargs):
        super(unireplknet_small, self).__init__(
            depths=[3, 3, 27, 3], embed_dim=96, drop_path_rate=0.3,
            kernel_sizes=None, with_cp=False, attempt_use_lk_impl=False, **kwargs)


class unireplknet_base(UniRepLKNet):
    def __init__(self, **kwargs):
        super(unireplknet_base, self).__init__(
            depths=[3, 3, 27, 3], embed_dim=128, drop_path_rate=0.3,
            kernel_sizes=None, with_cp=False, attempt_use_lk_impl=False, **kwargs)


class unireplknet_large(UniRepLKNet):
    def __init__(self, **kwargs):
        super(unireplknet_large, self).__init__(
            depths=[3, 3, 27, 3], embed_dim=192, drop_path_rate=0.4,
            kernel_sizes=None, with_cp=False, attempt_use_lk_impl=False, **kwargs)


class unireplknet_xlarge(UniRepLKNet):
    def __init__(self, **kwargs):
        super(unireplknet_xlarge, self).__init__(
            depths=[3, 3, 27, 3], embed_dim=256, drop_path_rate=0.4,
            kernel_sizes=None, with_cp=False, attempt_use_lk_impl=False, **kwargs)