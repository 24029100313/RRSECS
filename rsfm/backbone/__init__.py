# Design for Natural Data
from .resnet import *
from .vit import *
from .swin import *
from .mit import *
from .beit import *
from .pvtv2 import *
from .convnext import *
from .cswin import *
from .swinv2 import *
from .focalnet import *
from .iformer import *
from .internimage import *
from .van import *
from .unireplknet import *
from .transnext import *
from .inceptionnext import *
from .vmamba import *
from .casvit import *
from .dinov2 import *
from .mvitv2 import *
from .eva2 import *
from .wavevit import *
from .gfnet import *
from .starnet import *
from .moganet import *
from .mambaout import *
from .mlla import *
from .metaformer import *

# Design for Remote Sensing Data
from .lsknet import *
from .satmae_pp import *
from .scalemae import *
from .cross_scalemae import *
from .spectralgpt import *
from .hypersigma import *

# Methods designed by IPIU


__all__ = ['resnet50', 'resnet101',
           'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5',
           'pvtv2_b0', 'pvtv2_b1', 'pvtv2_b2', 'pvtv2_b3', 'pvtv2_b4', 'pvtv2_b5',
           'van_b0', 'van_b1', 'van_b2', 'van_b3', 'van_b4', 'van_b5', 'van_b6',
           'lsknet_tiny', 'lsknet_small',
           'casvit_xs', 'casvit_s', 'casvit_m', 'casvit_t',
           'swin_tiny', 'swin_small', 'swin_base', 'swin_large', 'swin_base_gfm_w6',
           'swinv2_tiny', 'swinv2_small', 'swinv2_base', 'swinv2_large',
           'cswin_tiny', 'cswin_small', 'cswin_base', 'cswin_large',
           'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge',
           'unireplknet_tiny', 'unireplknet_small', 'unireplknet_base', 'unireplknet_large', 'unireplknet_xlarge',
           'internimage_tiny', 'internimage_small', 'internimage_base', 'internimage_large', 'internimage_xlarge', 'internimage_huge',
           'focalnet_tiny', 'focalnet_small', 'focalnet_base', 'focalnet_large', 'focalnet_xlarge',
           'vmamba_tiny', 'vmamba_small', 'vmamba_base',
           'vit_base', 'vit_large',
           'dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant',
           'vit_large_patch16_satmae_pp_timm', 'vit_large_patch16_satmae_pp_rsfm',
           'vit_large_patch16_scalemae_timm', 'vit_large_patch16_scalemae_rsfm',
           'vit_large_patch16_cross_scalemae_timm', 'vit_large_patch16_cross_scalemae_rsfm',
           'vit_base_patch16_spectralgpt',
           'vit_base_patch16_spatsigma', 'vit_large_patch16_spatsigma', 'vit_huge_patch16_spatsigma',
           'vit_base_patch16_hypersigma', 'vit_large_patch16_hypersigma', 'vit_huge_patch16_hypersigma',
           'beit_base', 'beit_large', 'beitv2_base', 'beitv2_large',
           'iformer_small', 'iformer_base', 'iformer_large',
           'inceptionnext_tiny', 'inceptionnext_small', 'inceptionnext_base',
           'transnext_micro', 'transnext_tiny', 'transnext_small', 'transnext_base',
           'mvitv2_tiny', 'mvitv2_small', 'mvitv2_base', 'mvitv2_large', 'mvitv2_huge',
           'eva2_base', 'eva2_large',
           'wavevit_small', 'wavevit_base', 'wavevit_large',
           'gfnet_tiny', 'gfnet_small', 'gfnet_base',
           'starnet_s1', 'starnet_s2', 'starnet_s3', 'starnet_s4',
           'moganet_xtiny', 'moganet_tiny', 'moganet_small', 'moganet_base', 'moganet_large', 'moganet_xlarge',
           'mambaout_femto', 'mambaout_kobe', 'mambaout_tiny', 'mambaout_small', 'mambaout_base',
           'mlla_tiny', 'mlla_small', 'mlla_base',
           'convformer_s18', 'convformer_s36', 'convformer_b36', 'convformer_m36',
           'caformer_s18', 'caformer_s36', 'caformer_b36', 'caformer_m36',
           'poolformerv2_s12', 'poolformerv2_s24', 'poolformerv2_s36', 'poolformerv2_m36', 'poolformerv2_m48']