import torch
import torch.nn as nn
from .utils import MCN_fusion, MCN_decoder



class MCNHead(nn.Module):
    def __init__(self,
                 in_channels=[96, 192, 384, 768], 
                 embedding_dim=512, 
                 num_classes=2,
                 l_dim=768,
                 **kwargs):
        super(MCNHead, self).__init__()

        self.fusion = MCN_fusion(in_channels, l_dim, embedding_dim)
        self.decoder = MCN_decoder(embedding_dim, n_classes=num_classes)

    def forward(self, v, l, l_mask=None):
        det_attn, seg_attn = self.fusion(v, l) # det: B, C, H/32, W/32, seg: B, C, H/8, W/8
        output = self.decoder(det_attn, seg_attn)

        return output
