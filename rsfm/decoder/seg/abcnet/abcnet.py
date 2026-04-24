import torch.nn as nn
from .utils import ContextPath, SpatialPath, FeatureAggregationModule, Output


class ABCNetHead(nn.Module):
    '''
    from 'ABCNet: Attentive bilateral contextual network for efficient semantic segmentation of Fine-Resolution remotely sensed imagery, ISPRS 2021'
    '''
    def __init__(self, 
                 in_channels=[32, 64, 160, 256],
                 embedding_dim=256,
                 num_classes=7,
                 **kwargs):
        super(ABCNetHead, self).__init__()

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.cp = ContextPath(self.in_channels[2], self.in_channels[3])
        self.sp = SpatialPath()
        self.fam = FeatureAggregationModule(self.embedding_dim, self.embedding_dim)
        self.conv_out = Output(self.embedding_dim, self.embedding_dim, num_classes, up_factor=8)

    def forward(self, inputs, x):
        feat_cp8, feat_cp16 = self.cp(inputs)
        feat_sp = self.sp(x)
        feat_fuse = self.fam(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)

        return feat_out
