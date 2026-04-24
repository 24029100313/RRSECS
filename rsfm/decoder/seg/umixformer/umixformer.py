import torch.nn as nn
import torch
from collections import OrderedDict
from typing import List
from torch import Tensor
from rsfm.utils import resize
from .utils import Block, CatKey



class UMixFormerHead(nn.Module):
    """
    Attention-Pooling Former
    """
    def __init__(self,
                 num_classes=7,
                 in_index=[0, 1, 2, 3],
                 in_channels=[32, 64, 160, 256],
                 feature_strides=[4, 8, 16, 32],
                 embedding_dim=256,
                 dropout_ratio=0.1,
                 **kwargs):
        super(UMixFormerHead, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        self.in_index = in_index
        self.in_channels = in_channels
        self.feature_strides = feature_strides
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        tot_channels = sum(self.in_channels)

        self.attn_c4 = Block(dim1=c4_in_channels, dim2=tot_channels, num_heads=8, mlp_ratio=4, drop_path=0.1, pool_ratio=8)
        self.attn_c3 = Block(dim1=c3_in_channels, dim2=tot_channels, num_heads=4, mlp_ratio=4, drop_path=0.1, pool_ratio=4)
        self.attn_c2 = Block(dim1=c2_in_channels, dim2=tot_channels, num_heads=2, mlp_ratio=4, drop_path=0.1, pool_ratio=2)
        self.attn_c1 = Block(dim1=c1_in_channels, dim2=tot_channels, num_heads=1, mlp_ratio=4, drop_path=0.1, pool_ratio=1)

        self.cat_key1 = CatKey(pool_ratio=[1, 2, 4, 8], dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key2 = CatKey(pool_ratio=[1, 2, 4, 8], dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key3 = CatKey(pool_ratio=[1, 2, 4, 8], dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])
        self.cat_key4 = CatKey(pool_ratio=[1, 2, 4, 8], dim=[c4_in_channels, c3_in_channels, c2_in_channels, c1_in_channels])

        self.linear_fuse = nn.Sequential(
            OrderedDict(conv=nn.Conv2d(in_channels=tot_channels,
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
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        c_key = self.cat_key1([c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, channels]
        c4 = c4.flatten(2).transpose(1, 2)
        _c4 = self.attn_c4(c4, c_key, h4, w4, h4, w4)

        _c4 = _c4.permute(0,2,1).reshape(n, -1, h4, w4)
        c_key = self.cat_key2([_c4, c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, channels]
        c3 = c3.flatten(2).transpose(1, 2)
        _c3 = self.attn_c3(c3, c_key, h4, w4, h3, w3)

        _c3 = _c3.permute(0,2,1).reshape(n, -1, h3, w3)
        c_key = self.cat_key3([_c4, _c3, c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, channels]
        c2 = c2.flatten(2).transpose(1, 2)
        _c2 = self.attn_c2(c2, c_key, h4, w4, h2, w2)

        _c2 = _c2.permute(0,2,1).reshape(n, -1, h2, w2)
        c_key = self.cat_key4([_c4, _c3, _c2, c1])
        c_key = c_key.flatten(2).transpose(1, 2) #shape: [batch, h1*w1, channels]
        c1 = c1.flatten(2).transpose(1, 2)
        _c1 = self.attn_c1(c1, c_key, h4, w4, h1, w1)

        _c4 = resize(_c4, size=(h1,w1), mode='bilinear', align_corners=False)
        _c3 = resize(_c3, size=(h1,w1), mode='bilinear', align_corners=False)
        _c2 = resize(_c2, size=(h1,w1), mode='bilinear', align_corners=False)
        _c1 = _c1.permute(0,2,1).reshape(n, -1, h1, w1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
