from .eva2 import EVA2


__all__ = ['eva2_base', 'eva2_large']


class eva2_base(EVA2):
    def __init__(self, **kwargs):
        super(eva2_base, self).__init__(embed_dim=768, depth=12, num_heads=12,
                                        out_indices=[3, 5, 7, 11], drop_path_rate=0.15, **kwargs)


class eva2_large(EVA2):
    def __init__(self, **kwargs):
        super(eva2_large, self).__init__(embed_dim=1024, depth=24, num_heads=16,
                                         out_indices=[7, 11, 15, 23], drop_path_rate=0.2, **kwargs)