from .lavt_fusion import LAVT_fusion
from .rmsin_fusion import CIM, RMSIN_fusion
from .dmmi_fusion import DMMI_fusion
from .remamber_fusion import ReMamber_fusion
from .magnet_fusion import MagNet_fusion
from .refsegformer_fusion import RefSegformer_fusion
from .mct_fusion import MCT_fusion
from .crossvlt_fusion import CrossVLT_fusion
from .ccformer_fusion import CCFormer_fusion


__all__ = ['CIM', 'build_ris_fusion']


def build_ris_fusion(type, dim, l_dim, num_heads_fusion, fusion_drop, **kwargs):

    if type in ['LAVT', 'LGCE', 'CGFormer', 'ReLA']:
        return LAVT_fusion(dim, l_dim, num_heads_fusion, fusion_drop)
    elif type == 'RMSIN':
        return RMSIN_fusion(dim, l_dim, num_heads_fusion, fusion_drop)
    elif type == 'DMMI':
        return DMMI_fusion(dim, l_dim, num_heads_fusion, fusion_drop)
    elif type == 'MagNet':
        return MagNet_fusion(dim, l_dim, num_heads_fusion, fusion_drop)
    elif type == 'CrossVLT':
        return CrossVLT_fusion(dim, l_dim, num_heads_fusion, fusion_drop)
    elif type in ['CCFormer', 'CCFormerRIS']:
        return CCFormer_fusion(dim, l_dim, num_heads_fusion, fusion_drop)
    elif type == 'RefSegformer':
        return RefSegformer_fusion(dim, l_dim)
    elif type == 'ReMamber':
        return ReMamber_fusion(dim, l_dim)
    elif type == 'MCT':
        return MCT_fusion(dim, l_dim, **kwargs)
    else:
        raise NotImplementedError