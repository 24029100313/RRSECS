from torch import nn
from rsfm.utils import MLP
from ..vltvg.transformer import build_visual_encoder
from .decoder import build_bbox_regression


class TransCPHead(nn.Module):
    '''
    trans_enc: Bool
        True: original implementation, using a transformer encoder to enhance basic features from backbone
        False: using features from backbone directly
    '''
    def __init__(self,
                 img_size=640,
                 num_classes=1,
                 embedding_dim=256,
                 in_channels=[96, 192, 384, 768],
                 dropout=0.1,
                 num_heads=8,
                 num_enc_layer=6,
                 aux_loss=False,
                 trans_enc=True, **kwargs):
        super().__init__()
        self.aux_loss = aux_loss
        self.trans_enc = trans_enc

        if trans_enc:
            self.backbone_transformer = build_visual_encoder(embedding_dim=embedding_dim,
                                                             dropout=dropout,
                                                             num_heads=num_heads,
                                                             dim_feedforward=2048,
                                                             num_enc_layer=num_enc_layer)

        self.input_proj = nn.Conv2d(in_channels[-1], embedding_dim, kernel_size=1)

        self.text_proj = nn.Linear(768, embedding_dim)
        self.lstm_proj = nn.Linear(768, embedding_dim)

        if img_size == 640:
            num_vis_token = 400
        elif img_size == 512:
            num_vis_token = 256
        else:
            raise NotImplementedError

        self.bbox_regression = build_bbox_regression(num_vis_token, aux_loss)
        self.bbox_embed = MLP(embedding_dim, embedding_dim, 4, 3)

    def forward(self, features, pos, text_feat, text_mask):
        # text_feat: disentangled_lang, text_mask: bert_fea

        vis_feat, vis_mask = features[-1].decompose()
        bs, c, h, w = vis_feat.size()

        if self.trans_enc:
            vis_feat, vis_mask, pos_embed = self.backbone_transformer(self.input_proj(vis_feat), vis_mask, pos[-1])
        else:
            vis_feat = self.input_proj(vis_feat).flatten(2).permute(2, 0, 1)
            vis_mask = vis_mask.flatten(1)
            pos_embed = pos[-1].flatten(2).permute(2, 0, 1)

        disentangled_lang, bert_fea = text_feat, text_mask
        word_feat, word_mask = bert_fea.decompose()
        word_feat = self.text_proj(word_feat)
        word_feat = word_feat.permute(1, 0, 2)  # NxLxC -> LxNxC
        word_mask = word_mask.flatten(1)
        projected_disentangled_lang = self.lstm_proj(disentangled_lang).squeeze().unsqueeze(-1).unsqueeze(-1)

        hs = self.bbox_regression(vis_feat, vis_mask, pos_embed, word_feat, word_mask, projected_disentangled_lang, h, w)

        if self.aux_loss:
            out_list = []
            for aux_pred in hs:
                out_list.append(self.bbox_embed(aux_pred[0]).sigmoid())

            if self.training:
                return out_list

            return out_list[-1]

        pred_box = self.bbox_embed(hs[-1][0]).sigmoid()  # [B, 4]

        return pred_box