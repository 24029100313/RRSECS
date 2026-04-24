# Design for Natural Data
from .segformer import SegFormerHead
from .umixformer import UMixFormerHead
from .deeplabv3plus import DeepLabV3PlusHead
from .lightham import LightHamHead
from .upernet import UPerHead
from .semantic_fpn import FPNHead
from .mask2former import Mask2FormerHead
from .dpt import DPTHead

# Design for Remote Sensing Data
from .abcnet import ABCNetHead
from .aerialformer import AerialFormerHead
from .banet import BANetHead
from .dcswin import DCSwinHead
from .unetformer import UNetFormerHead

# Design for Medical Data
from .unet import UNetHead
from .duat import DuATHead
from .emcad import EMCADHead
from .unetv2 import UNetv2Head
from .unetpp import UNetPPHead
from .attnunet import AttnUNetHead
from .cascade import CASCADEHead
from .polyp import PolypHead

# Methods designed by IPIU


__all__ = ['SegFormerHead', 'UMixFormerHead', 'UNetFormerHead', 'UNetv2Head', 'DeepLabV3PlusHead', 'DPTHead',
           'LightHamHead', 'UPerHead', 'FPNHead', 'UNetHead', 'Mask2FormerHead', 'EMCADHead', 'UNetPPHead',
           'ABCNetHead', 'AerialFormerHead', 'BANetHead', 'DCSwinHead', 'DuATHead', 'AttnUNetHead', 'CASCADEHead', 'PolypHead']