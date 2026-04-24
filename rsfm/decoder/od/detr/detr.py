from torch import nn
from rsfm.utils import MLP
import torch
from .transformer import build_transformer



class DETRHead(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self,
                 embedding_dim=256,
                 in_channels=[96, 192, 384, 768],
                 num_classes=10,
                 num_queries=100,
                 num_enc_layer=6,
                 num_dec_layer=6,
                 dropout=0.1,
                 num_heads=8,
                 aux_loss=False, **kwargs):
        super().__init__()

        self.transformer = build_transformer(embedding_dim=embedding_dim,
                                             dropout=dropout,
                                             num_heads=num_heads,
                                             dim_feedforward=2048,
                                             enc_layers=num_enc_layer,
                                             dec_layers=num_dec_layer)

        self.num_queries = num_queries
        hidden_dim = embedding_dim
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(in_channels[-1], hidden_dim, kernel_size=1)

        self.aux_loss = aux_loss

    def forward(self, features, pos):
        # only use the last layer feature
        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]