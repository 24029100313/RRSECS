from .beit import BEiT


__all__ = ['beit_base', 'beit_large',
           'beitv2_base', 'beitv2_large']


class beit_base(BEiT):
    def __init__(self, **kwargs):
        super(beit_base, self).__init__(
            use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, drop_path_rate=0.1, **kwargs)


class beit_large(BEiT):
    def __init__(self, **kwargs):
        super(beit_large, self).__init__(
            embed_dim=1024, depth=24, num_heads=16, use_abs_pos_emb=False, use_rel_pos_bias=True,
            init_values=1e-6, drop_path_rate=0.2, out_indices=[7, 11, 15, 23], **kwargs)


class beitv2_base(BEiT):
    def __init__(self, **kwargs):
        super(beitv2_base, self).__init__(
            use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, drop_path_rate=0.15, **kwargs)


class beitv2_large(BEiT):
    def __init__(self, **kwargs):
        super(beitv2_large, self).__init__(
            embed_dim=1024, depth=24, num_heads=16, use_abs_pos_emb=False, use_rel_pos_bias=True,
            init_values=1e-6, drop_path_rate=0.2, out_indices=[7, 11, 15, 23], **kwargs)