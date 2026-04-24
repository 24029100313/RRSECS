import torch.nn as nn
from torch import Tensor
from typing import List
from .utils import ConvBN, Block, WF, FeatureRefinementHead, ConvBNReLU, Conv



class UNetFormerHead(nn.Module):
    def __init__(self,
                 num_classes=7,
                 in_index=[0, 1, 2, 3],
                 in_channels=[64, 128, 256, 512],
                 embedding_dim=64,
                 dropout_ratio=0.1,
                 window_size=8,
                 **kwargs):
        super(UNetFormerHead, self).__init__()
        self.in_index = in_index
        
        self.pre_conv = ConvBN(in_channels[-1], embedding_dim, kernel_size=1)
        self.b4 = Block(dim=embedding_dim, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=embedding_dim, num_heads=8, window_size=window_size)
        self.p3 = WF(in_channels[-2], embedding_dim)

        self.b2 = Block(dim=embedding_dim, num_heads=8, window_size=window_size)
        self.p2 = WF(in_channels[-3], embedding_dim)

        self.p1 = FeatureRefinementHead(in_channels[-4], embedding_dim)

        self.segmentation_head = nn.Sequential(ConvBNReLU(embedding_dim, embedding_dim),
                                               nn.Dropout2d(p=dropout_ratio, inplace=True),
                                               Conv(embedding_dim, num_classes, kernel_size=1))
        self.init_weight()


    def forward(self, inputs:List[Tensor]):
        x = [inputs[i] for i in self.in_index]  # len=4, 1/4,1/8,1/16,1/32
        res1, res2, res3, res4 = x

        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)
        x = self.p2(x, res2)
        x = self.b2(x)
        x = self.p1(x, res1)
        x = self.segmentation_head(x)

        return x


    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
