import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List
from collections import OrderedDict
from .utils import MLP



class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self,
                 in_channels=[32, 64, 160, 256],
                 in_index=[0, 1, 2, 3],
                 feature_strides=[4, 8, 16, 32],
                 dropout_ratio=0.1,
                 num_classes=7,
                 embedding_dim=256,
                 **kwargs):
        super(SegFormerHead, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        self.in_index = in_index
        self.in_channels = in_channels
        self.feature_strides = feature_strides
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # TODO: check other default args
        self.linear_fuse = nn.Sequential(
            OrderedDict(conv=nn.Conv2d(in_channels=embedding_dim*4,
                                       out_channels=embedding_dim,
                                       kernel_size=1,
                                       bias=False),
                        bn=nn.BatchNorm2d(num_features=embedding_dim),
                        act=nn.ReLU(inplace=True)
            )
        )

        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs:List[Tensor]):
        x = [inputs[i] for i in self.in_index]  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        # https://github.com/NVlabs/SegFormer/blob/master/mmseg/ops/wrappers.py
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

