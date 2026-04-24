from .seg import predict_seg
from .cd import predict_cd
from .cls import predict_cls
from .ris import predict_ris
from .vg import predict_vg
from .recs import predict_recs


__all__ = ['predict_seg', 'predict_cd', 'predict_cls', 'predict_ris', 'predict_vg', 'predict_recs']