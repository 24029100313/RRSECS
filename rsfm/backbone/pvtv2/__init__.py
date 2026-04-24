from torch import nn
from functools import partial
from .pvtv2 import PyramidVisionTransformerV2


__all__ = ['pvtv2_b0', 'pvtv2_b1', 'pvtv2_b2', 'pvtv2_b3', 'pvtv2_b4', 'pvtv2_b5']


class pvtv2_b0(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvtv2_b0, self).__init__(
            embed_dim=32, num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)

class pvtv2_b1(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvtv2_b1, self).__init__(
            embed_dim=64, num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)

class pvtv2_b2(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvtv2_b2, self).__init__(
            embed_dim=64, num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)

class pvtv2_b3(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvtv2_b3, self).__init__(
            embed_dim=64, num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)

class pvtv2_b4(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvtv2_b4, self).__init__(
            embed_dim=64, num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)

class pvtv2_b5(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvtv2_b5, self).__init__(
            embed_dim=64, num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)