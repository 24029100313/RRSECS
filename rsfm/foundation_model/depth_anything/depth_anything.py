import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_state_dict
from .utils import DPT, get_encoder


class DepthAnything(nn.Module):
    def __init__(self, type='small', img_size=518):
        super(DepthAnything, self).__init__()

        encoder, embed_dim, features = get_encoder(type, img_size)

        self.pretrained = encoder

        self.depth_head = DPT(in_channels=[embed_dim // 2 ** (i - 1)
                                           for i in range(4, 0, -1)] if type != 'large' else [256, 512, 1024, 1024],
                              embedding_dim=features)

    def init_weights(self, pretrained=None):
        assert pretrained is not None, 'You must set it !!!'

        checkpoint = torch.load(pretrained, map_location='cpu')
        load_state_dict(self, checkpoint, False)

    def forward(self, x):
        h, w = x.shape[-2:]

        features = self.pretrained(x)

        depth = self.depth_head(features)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)
