from torch import nn
from .lsknet import LSKNet
from functools import partial



class lsknet_tiny(LSKNet):
    def __init__(self, **kwargs):
        super(lsknet_tiny, self).__init__(
            embed_dim=32, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2], **kwargs)


class lsknet_small(LSKNet):
    def __init__(self, **kwargs):
        super(lsknet_small, self).__init__(
            embed_dim=64, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 4, 2], **kwargs)