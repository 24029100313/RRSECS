import torch.nn as nn
from torch import Tensor
from typing import List
import numpy as np
from rsfm.module.neck.utils import CBR
from rsfm.utils import resize
from .utils import Upsample



class FPNHead(nn.Module):
    def __init__(self,
                 in_channels=[256, 256, 256, 256],
                 in_index=[0, 1, 2, 3],
                 feature_strides=[4, 8, 16, 32],
                 dropout_ratio=0.1,
                 num_classes=7,
                 embedding_dim=128,
                 **kwargs):
        super(FPNHead, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.in_index = in_index
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio

        self.scale_heads = nn.ModuleList()

        for i in range(len(feature_strides)):
            head_length = max(1, int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    CBR(self.in_channels[i] if k == 0 else self.embedding_dim, self.embedding_dim, 3, 1, 1))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            self.scale_heads.append(nn.Sequential(*scale_head))

        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs:List[Tensor]):
        x = [inputs[i] for i in self.in_index]

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=False)

        output = self.dropout(output)
        output = self.linear_pred(output)
        return output
