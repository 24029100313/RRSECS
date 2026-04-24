from .qrnet_fusion import QRNet_fusion
from .lpva_fusion import LPVA_fusion


__all__ = ['build_vg_fusion']


def build_vg_fusion(type, dim, l_dim, **kwargs):
    if type == 'QRNet':
        return QRNet_fusion(dim, l_dim)
    elif type == 'LPVA':
        return LPVA_fusion(dim, l_dim, **kwargs)
    else:
        raise NotImplementedError