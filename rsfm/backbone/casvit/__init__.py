from .casvit import CASViT


__all__ = ['casvit_xs', 'casvit_s', 'casvit_m', 'casvit_t']


class casvit_xs(CASViT):
    def __init__(self, **kwargs):
        super(casvit_xs, self).__init__(layers=[2, 2, 4, 2], embed_dims=[48, 56, 112, 220], **kwargs)


class casvit_s(CASViT):
    def __init__(self, **kwargs):
        super(casvit_s, self).__init__(layers=[3, 3, 6, 3], embed_dims=[48, 64, 128, 256], **kwargs)


class casvit_m(CASViT):
    def __init__(self, **kwargs):
        super(casvit_m, self).__init__(layers=[3, 3, 6, 3], embed_dims=[64, 96, 192, 384], **kwargs)


class casvit_t(CASViT):
    def __init__(self, **kwargs):
        super(casvit_t, self).__init__(layers=[3, 3, 6, 3], embed_dims=[96, 128, 256, 512], **kwargs)
