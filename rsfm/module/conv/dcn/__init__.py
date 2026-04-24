from .dcn import DCN


__all__ = ['DCN', 'DCNv2']


class DCNv2(DCN):
    def __init__(self, **kwargs):
        super(DCNv2, self).__init__(modulation=True, **kwargs)