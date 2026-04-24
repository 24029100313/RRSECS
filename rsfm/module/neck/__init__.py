from .fpn import FPN
from .bifpn import BiFPN
from .pafpn import PAFPN
from .multilevel_neck import MultiLevelNeck
from .qmf import QMF
from .mscab import MSCAB
from .vg_neck import vg_neck
from .p2h import P2H


__all__ = ['fpn', 'pafpn', 'bifpn', 'multilevel_neck', 'QMF', 'MSCAB', 'vg_neck', 'p2h']


class fpn(FPN):
    def __init__(self, **kwargs):
        super(fpn, self).__init__(**kwargs)


class pafpn(PAFPN):
    def __init__(self, **kwargs):
        super(pafpn, self).__init__(**kwargs)


class bifpn(BiFPN):
    def __init__(self, **kwargs):
        super(bifpn, self).__init__(**kwargs)


class multilevel_neck(MultiLevelNeck):
    def __init__(self, **kwargs):
        super(multilevel_neck, self).__init__(**kwargs)


class p2h(P2H):
    def __init__(self, **kwargs):
        super(p2h, self).__init__(**kwargs)