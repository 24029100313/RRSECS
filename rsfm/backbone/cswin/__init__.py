from .cswin import CSWin


__all__ = ['cswin_tiny', 'cswin_small', 'cswin_base', 'cswin_large']



class cswin_tiny(CSWin):
    def __init__(self, **kwargs):
        super(cswin_tiny, self).__init__(
            embed_dim=64, depth=[1, 2, 21, 1], num_heads=[2, 4, 8, 16], drop_path_rate=0.3, **kwargs)


class cswin_small(CSWin):
    def __init__(self, **kwargs):
        super(cswin_small, self).__init__(
            embed_dim=64, depth=[2, 4, 32, 2], num_heads=[2, 4, 8, 16], drop_path_rate=0.4, **kwargs)


class cswin_base(CSWin):
    def __init__(self, **kwargs):
        super(cswin_base, self).__init__(
            embed_dim=96, depth=[2, 4, 32, 2], num_heads=[4, 8, 16, 32], drop_path_rate=0.6, **kwargs)


class cswin_large(CSWin):
    def __init__(self, **kwargs):
        super(cswin_large, self).__init__(
            embed_dim=144, depth=[2, 4, 32, 2], num_heads=[6, 12, 24, 48], drop_path_rate=0.7, **kwargs)