import torch.nn as nn
from .utils import CBR, Attention_Embedding, TexturePath, FeatureAggregationModule, Output


class BANetHead(nn.Module):
    '''
    from 'Transformer meets convolution: A bilateral awareness network for semantic segmentation of very fine resolution urban scene images, RS 2021'
    '''
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 num_classes=7,
                 embedding_dim=256,
                 **kwargs):
        super(BANetHead, self).__init__()

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim 

        self.AE = Attention_Embedding(self.in_channels[3], self.in_channels[2])
        self.conv_avg = CBR(self.in_channels[2], 128, 1)
        self.up = nn.Upsample(scale_factor=2.)

        self.sp = TexturePath()
        self.fam = FeatureAggregationModule(self.embedding_dim, self.embedding_dim)
        self.conv_out = Output(self.embedding_dim, self.embedding_dim, num_classes, up_factor=8)

    def forward(self, inputs, x):

        e3 = inputs[-2]
        e4 = inputs[-1]
        e = self.conv_avg(self.AE(e4, e3))
        feat = self.up(e)

        feat_sp = self.sp(x)
        feat_fuse = self.fam(feat_sp, feat)

        feat_out = self.conv_out(feat_fuse)

        return feat_out
