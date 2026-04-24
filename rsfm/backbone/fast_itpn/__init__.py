from .fast_itpn import Fast_iTPN


__all__ = ['fast_itpn_tiny', 'fast_itpn_small', 'fast_itpn_base', 'fast_itpn_large']


class fast_itpn_tiny(Fast_iTPN):
    def __init__(self, **kwargs):
        super(fast_itpn_tiny, self).__init__(
            embed_dim=384, depth_stage1=1, depth_stage2=1, depth=12, num_heads=6, drop_path_rate=0.1, **kwargs)


class fast_itpn_small(Fast_iTPN):
    def __init__(self, **kwargs):
        super(fast_itpn_small, self).__init__(
            embed_dim=384, depth_stage1=2, depth_stage2=2, depth=20, num_heads=6, drop_path_rate=0.1, **kwargs)


class fast_itpn_base(Fast_iTPN):
    def __init__(self, **kwargs):
        super(fast_itpn_base, self).__init__(
            embed_dim=512, depth_stage1=3, depth_stage2=3, depth=24, num_heads=8, drop_path_rate=0.15, **kwargs)


class fast_itpn_large(Fast_iTPN):
    def __init__(self, **kwargs):
        super(fast_itpn_large, self).__init__(
            embed_dim=768, depth_stage1=2, depth_stage2=2, depth=40, num_heads=12, drop_path_rate=0.2, **kwargs)