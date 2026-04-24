from .dysample import DySample


__all__ = ['dysample', 'dysampleplus', 'dysample_s', 'dysample_splus']


class dysample(DySample):
    def __init__(self, **kwargs):
        super(dysample, self).__init__(style='lp', dyscope=False, **kwargs)


class dysampleplus(DySample):
    def __init__(self, **kwargs):
        super(dysampleplus, self).__init__(style='lp', dyscope=True, **kwargs)


class dysample_s(DySample):
    def __init__(self, **kwargs):
        super(dysample_s, self).__init__(style='pl', dyscope=False, **kwargs)


class dysample_splus(DySample):
    def __init__(self, **kwargs):
        super(dysample_splus, self).__init__(style='pl', dyscope=True, **kwargs)