from .iformer import InceptionTransformer


__all__ = ['iformer_small', 'iformer_base', 'iformer_large']


class iformer_small(InceptionTransformer):
    def __init__(self, **kwargs):
        super(iformer_small, self).__init__(
            depths=[3, 3, 9, 3], embed_dims=[96, 192, 320, 384], num_heads=[3, 6, 10, 12],
            attention_heads=[1] * 3 + [3] * 3 + [7] * 4 + [9] * 5 + [11] * 3,
            use_layer_scale=True, layer_scale_init_value=1e-6, **kwargs)


class iformer_base(InceptionTransformer):
    def __init__(self, **kwargs):
        super(iformer_base, self).__init__(
            depths=[4, 6, 14, 6], embed_dims=[96, 192, 384, 512], num_heads=[3, 6, 12, 16],
            attention_heads=[1] * 4 + [3] * 6 + [8] * 7 + [10] * 7 + [15] * 6,
            use_layer_scale=True, layer_scale_init_value=1e-6, **kwargs)


class iformer_large(InceptionTransformer):
    def __init__(self, **kwargs):
        super(iformer_large, self).__init__(
            depths=[4, 6, 18, 8], embed_dims=[96, 192, 448, 640], num_heads=[3, 6, 14, 20],
            attention_heads=[1] * 4 + [3] * 6 + [10] * 9 + [12] * 9 + [19] * 8,
            use_layer_scale=True, layer_scale_init_value=1e-6, **kwargs)