import torch
from torch import nn
from .utils import GLSA, SBA, Upsample, CBR
import torch.nn.functional as F

            
class DuATHead(nn.Module):
    '''
    from 'DuAT: Dual-Aggregation Transformer Network for Medical Image Segmentation, PRCV 2023'
    '''
    def __init__(self,
                 num_classes=7,
                 in_channels=[96, 192, 384, 768],
                 embedding_dim=64,
                 **kwargs):
        super(DuATHead, self).__init__()

        self.GLSA_c4 = GLSA(input_dim=in_channels[3], embed_dim=embedding_dim)
        self.GLSA_c3 = GLSA(input_dim=in_channels[2], embed_dim=embedding_dim)
        self.GLSA_c2 = GLSA(input_dim=in_channels[1], embed_dim=embedding_dim)

        self.L_feature = CBR(in_channels[0], embedding_dim, 3, 1, 1)
        self.SBA = SBA(input_dim = embedding_dim, output_dim = num_classes)

        self.fuse = CBR(embedding_dim * 2, embedding_dim, 1)
        self.fuse2 = nn.Sequential(CBR(embedding_dim*3, embedding_dim, 1),
                                   nn.Conv2d(embedding_dim, num_classes, kernel_size=1, bias=False))

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape
        _c4 = self.GLSA_c4(c4) # 1/32
        _c4 = Upsample(_c4, c3.size()[2:]) # 1/16
        _c3 = self.GLSA_c3(c3)
        _c2 = self.GLSA_c2(c2)

        output = self.fuse2(torch.cat([Upsample(_c4, c2.size()[2:]), Upsample(_c3, c2.size()[2:]), _c2], dim=1))

        L_feature = self.L_feature(c1)
        H_feature = self.fuse(torch.cat([_c4, _c3], dim=1))
        H_feature = Upsample(H_feature, c2.size()[2:])

        output2 = self.SBA(H_feature, L_feature)

        output = F.interpolate(output, scale_factor=8, mode='bilinear')
        output2 = F.interpolate(output2, scale_factor=4, mode='bilinear')

        return output + output2
