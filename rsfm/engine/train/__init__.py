from .seg import train_seg
from .cd import train_cd
from .cls import train_cls
from .ris import train_ris
from .vg import train_vg
from .od import train_od
from .recs import train_recs


__all__ = ['train_seg', 'train_cd', 'train_cls',
           'train_ris', 'train_vg', 'train_od', 'train_recs']