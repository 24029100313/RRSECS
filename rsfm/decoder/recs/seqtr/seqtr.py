import torch
import torch.nn as nn
from .utils import SimpleFusion, SeqHead



class SeqTRHead(nn.Module):
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 num_classes=2,
                 l_dim=768,
                 embedding_dim=256,
                 num_bin=1000,
                 num_ray=18,
                 **kwargs):
        super(SeqTRHead, self).__init__()

        self.fusion = SimpleFusion(vis_chs=in_channels[1:], direction='bottom_up', l_dim=l_dim)

        self.head = SeqHead(in_ch=in_channels[-1],
                            num_bin=num_bin,
                            multi_task=True,
                            shuffle_fraction=-1,
                            mapping='relative',
                            top_p=-1,
                            num_ray=num_ray,
                            det_coord=[0],
                            det_coord_weight=1.5,
                            predictor=dict(
                                num_fcs=3,
                                in_chs=[embedding_dim, embedding_dim, embedding_dim],
                                out_chs=[embedding_dim, embedding_dim, num_bin+1],
                                fc=[
                                    dict(
                                        linear=dict(type='Linear', bias=True),
                                        act=dict(type='ReLU', inplace=True),
                                        drop=None
                                    ),
                                    dict(
                                        linear=dict(type='Linear', bias=True),
                                        act=dict(type='ReLU', inplace=True),
                                        drop=None
                                    ),
                                    dict(
                                        linear=dict(type='Linear', bias=True),
                                        act=None,
                                        drop=None
                                    )
                                ]
                            ),
                            transformer=dict(
                                type='AutoRegressiveTransformer',
                                encoder=dict(
                                    num_layers=6,
                                    layer=dict(
                                        d_model=embedding_dim,
                                        nhead=8,
                                        dim_feedforward=4*embedding_dim,
                                        dropout=0.1,
                                        activation='relu',
                                        batch_first=True)),
                                decoder=dict(
                                    num_layers=3,
                                    layer=dict(
                                        d_model=embedding_dim,
                                        nhead=8,
                                        dim_feedforward=4*embedding_dim,
                                        dropout=0.1,
                                        activation='relu',
                                        batch_first=True),
                                )),
                            x_positional_encoding=dict(
                                type='SinePositionalEncoding2D',
                                num_feature=embedding_dim//2,
                                normalize=True),
                            seq_positional_encoding=dict(
                                type='LearnedPositionalEncoding1D',
                                num_embedding=1+4+1+2*num_ray,
                                num_feature=embedding_dim)
                            )

    def forward(self, features, x_mask, text_feat, text_mask):
        x_mm = self.fusion(features[1:], text_feat.mean(dim=-1))
        out = self.head(x_mm, x_mask)

        return x_mm

