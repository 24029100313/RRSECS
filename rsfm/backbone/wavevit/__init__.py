from .wavevit import WaveViT


__all__ = ['wavevit_small', 'wavevit_base', 'wavevit_large']


class wavevit_small(WaveViT):
    def __init__(self, **kwargs):
        super(wavevit_small, self).__init__(
            stem_hidden_dim=32, embed_dims=[64, 128, 320, 448], num_heads=[2, 4, 10, 14],
            depths=[3, 4, 6, 3], drop_path_rate=0.1, with_cp=False, **kwargs)


class wavevit_base(WaveViT):
    def __init__(self, **kwargs):
        super(wavevit_base, self).__init__(
            stem_hidden_dim=64, embed_dims=[64, 128, 320, 512], num_heads=[2, 4, 10, 16],
            depths=[3, 4, 12, 3], drop_path_rate=0.2, with_cp=False, **kwargs)


class wavevit_large(WaveViT):
    def __init__(self, **kwargs):
        super(wavevit_large, self).__init__(
            stem_hidden_dim=64, embed_dims=[96, 192, 384, 512], num_heads=[3, 6, 12, 16],
            depths=[3, 6, 18, 3], drop_path_rate=0.3, with_cp=False, **kwargs)