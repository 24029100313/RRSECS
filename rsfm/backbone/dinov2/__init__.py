from .dinov2 import DINOv2


__all__ = ['dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant']


class dinov2_small(DINOv2):
    def __init__(self, **kwargs):
        super(dinov2_small, self).__init__(
            patch_size=14, embed_dim=384, depth=12, out_indices=[2, 5, 8, 11], num_heads=6, **kwargs)


class dinov2_base(DINOv2):
    def __init__(self, **kwargs):
        super(dinov2_base, self).__init__(
            patch_size=14, embed_dim=768, depth=12, out_indices=[2, 5, 8, 11], num_heads=12, **kwargs)


class dinov2_large(DINOv2):
    def __init__(self, **kwargs):
        super(dinov2_large, self).__init__(
            patch_size=14, embed_dim=1024, depth=24, out_indices=[4, 11, 17, 23], num_heads=16, **kwargs)


class dinov2_giant(DINOv2):
    def __init__(self, **kwargs):
        super(dinov2_giant, self).__init__(
            patch_size=14, embed_dim=1536, depth=40, out_indices=[9, 19, 29, 39],
            num_heads=24, ffn_layer="swiglufused", **kwargs)