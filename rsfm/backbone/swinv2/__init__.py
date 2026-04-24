from .swinv2 import SwinTransformerV2


__all__ = ['swinv2_tiny', 'swinv2_small', 'swinv2_base', 'swinv2_large']



class swinv2_tiny(SwinTransformerV2):
    def __init__(self, **kwargs):
        super(swinv2_tiny, self).__init__(
            embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], drop_path_rate=0.2, **kwargs)


class swinv2_small(SwinTransformerV2):
    def __init__(self, **kwargs):
        super(swinv2_small, self).__init__(
            embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], drop_path_rate=0.3, **kwargs)


class swinv2_base(SwinTransformerV2):
    def __init__(self, **kwargs):
        super(swinv2_base, self).__init__(
            embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
            drop_path_rate=0.2, pretrained_window_size=[12, 12, 12, 6], **kwargs)


class swinv2_large(SwinTransformerV2):
    def __init__(self, **kwargs):
        super(swinv2_large, self).__init__(
            embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
            drop_path_rate=0.2, pretrained_window_size=[12, 12, 12, 6], **kwargs)
