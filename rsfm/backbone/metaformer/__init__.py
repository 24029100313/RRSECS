from .metaformer import MetaFormer
from .utils import SepConv, Attention, Pooling, LayerNormGeneral, partial


__all__ = ['convformer_s18', 'convformer_s36', 'convformer_b36', 'convformer_m36',
           'caformer_s18', 'caformer_s36', 'caformer_b36', 'caformer_m36',
           'poolformerv2_s12', 'poolformerv2_s24', 'poolformerv2_s36', 'poolformerv2_m36', 'poolformerv2_m48']


class convformer_s18(MetaFormer):
    def __init__(self, **kwargs):
        super(convformer_s18, self).__init__(
            depths=[3, 3, 9, 3], embed_dims=[64, 128, 320, 512], drop_path_rate=0.1, token_mixers=SepConv, **kwargs)


class convformer_s36(MetaFormer):
    def __init__(self, **kwargs):
        super(convformer_s36, self).__init__(
            depths=[3, 12, 18, 3], embed_dims=[64, 128, 320, 512], drop_path_rate=0.2, token_mixers=SepConv, **kwargs)


class convformer_m36(MetaFormer):
    def __init__(self, **kwargs):
        super(convformer_m36, self).__init__(
            depths=[3, 12, 18, 3], embed_dims=[96, 192, 384, 576], drop_path_rate=0.3, token_mixers=SepConv, **kwargs)


class convformer_b36(MetaFormer):
    def __init__(self, **kwargs):
        super(convformer_b36, self).__init__(
            depths=[3, 12, 18, 3], embed_dims=[128, 256, 512, 768], drop_path_rate=0.4, token_mixers=SepConv, **kwargs)


class caformer_s18(MetaFormer):
    def __init__(self, **kwargs):
        super(caformer_s18, self).__init__(depths=[3, 3, 9, 3], embed_dims=[64, 128, 320, 512], drop_path_rate=0.1,
                                           token_mixers=[SepConv, SepConv, Attention, Attention], **kwargs)


class caformer_s36(MetaFormer):
    def __init__(self, **kwargs):
        super(caformer_s36, self).__init__(depths=[3, 12, 18, 3], embed_dims=[64, 128, 320, 512], drop_path_rate=0.2,
                                           token_mixers=[SepConv, SepConv, Attention, Attention], **kwargs)


class caformer_m36(MetaFormer):
    def __init__(self, **kwargs):
        super(caformer_m36, self).__init__(depths=[3, 12, 18, 3], embed_dims=[96, 192, 384, 576], drop_path_rate=0.3,
                                           token_mixers=[SepConv, SepConv, Attention, Attention], **kwargs)


class caformer_b36(MetaFormer):
    def __init__(self, **kwargs):
        super(caformer_b36, self).__init__(depths=[3, 12, 18, 3], embed_dims=[128, 256, 512, 768], drop_path_rate=0.4,
                                           token_mixers=[SepConv, SepConv, Attention, Attention], **kwargs)


class poolformerv2_s12(MetaFormer):
    def __init__(self, **kwargs):
        super(poolformerv2_s12, self).__init__(
            depths=[2, 2, 6, 2], embed_dims=[64, 128, 320, 512], drop_path_rate=0.1, token_mixers=Pooling,
            norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False), **kwargs)


class poolformerv2_s24(MetaFormer):
    def __init__(self, **kwargs):
        super(poolformerv2_s24, self).__init__(
            depths=[4, 4, 12, 4], embed_dims=[64, 128, 320, 512], drop_path_rate=0.1, token_mixers=Pooling,
            norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False), **kwargs)


class poolformerv2_s36(MetaFormer):
    def __init__(self, **kwargs):
        super(poolformerv2_s36, self).__init__(
            depths=[6, 6, 18, 6], embed_dims=[64, 128, 320, 512], drop_path_rate=0.2, token_mixers=Pooling,
            norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False), **kwargs)


class poolformerv2_m36(MetaFormer):
    def __init__(self, **kwargs):
        super(poolformerv2_m36, self).__init__(
            depths=[6, 6, 18, 6], embed_dims=[96, 192, 384, 768], drop_path_rate=0.3, token_mixers=Pooling,
            norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False), **kwargs)


class poolformerv2_m48(MetaFormer):
    def __init__(self, **kwargs):
        super(poolformerv2_m48, self).__init__(
            depths=[6, 6, 18, 6], embed_dims=[96, 192, 384, 768], drop_path_rate=0.4, token_mixers=Pooling,
            norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False), **kwargs)