import torch
import torch.nn as nn
from rsfm.utils import MLP
from .transformer import TransformerDecoderLayer_self, TransformerDecoderLayer_self_2
from rsfm.decoder.vg.transvg.vl_transformer import build_vl_transformer
from rsfm.decoder.vg.transvg.transformer import build_transformer



class LPVAHead(nn.Module):
    '''
    from 'Language-Guided Progressive Attention for Visual Grounding in Remote Sensing Images, TGRS 2024'
    '''

    def __init__(self,
                 img_size=640,
                 embedding_dim=256,
                 in_channels=[96, 192, 384, 768],
                 dropout=0.1,
                 num_heads=8,
                 num_enc_layer=6,
                 num_classes=1,
                 aux_loss=False,
                 trans_enc=False,
                 **kwargs):
        super(LPVAHead, self).__init__()
        
        self.num_vis_token = int((img_size / 32) ** 2) # (640 / 32)^2 = 400
        self.num_text_token = 20
        self.aux_loss = aux_loss
        self.trans_enc = trans_enc

        if trans_enc:
            self.backbone_transformer = build_transformer(embedding_dim=embedding_dim,
                                                          dropout=dropout,
                                                          num_heads=num_heads,
                                                          dim_feedforward=2048,
                                                          enc_layers=num_enc_layer,
                                                          dec_layers=0)

            self.input_proj = nn.Conv2d(in_channels[-1], embedding_dim, kernel_size=1)

        self.mhead1 = TransformerDecoderLayer_self(embedding_dim, 8)
        self.mhead2 = TransformerDecoderLayer_self(embedding_dim, 8)
        self.mhead3 = TransformerDecoderLayer_self_2(embedding_dim, 8)

        num_total = self.num_vis_token + self.num_text_token + 1
        self.reg_token = nn.Embedding(1, embedding_dim)
        self.vl_pos_embed = nn.Embedding(num_total, embedding_dim)

        self.vis_proj = nn.Linear(in_channels[-1], embedding_dim)
        self.text_proj = nn.Linear(768, embedding_dim)

        self.vl_transformer = build_vl_transformer(embedding_dim=embedding_dim,
                                                   dropout_rate=dropout,
                                                   num_heads=num_heads,
                                                   dim_feedforward=2048,
                                                   num_enc_layer=num_enc_layer,
                                                   return_intermediate=aux_loss)

        self.bbox_embed = MLP(embedding_dim, embedding_dim, 4, 3)


    def forward(self, features, pos, text_feat, text_mask):
        vis_feat, vis_mask = features[-1].decompose()
        B = vis_feat.shape[0]

        if self.trans_enc:
            vis_mask, vis_feat = self.backbone_transformer(self.input_proj(vis_feat), vis_mask,
                                                           pos[-1], query_embed=None)
        else:
            vis_mask, vis_feat = vis_mask.flatten(1), vis_feat.flatten(2).permute(2, 0, 1)

        vis_feat = self.vis_proj(vis_feat)  # [HW/32*32, B, embedding_dim]

        text_feat = self.text_proj(text_feat).permute(1, 0, 2)
        text_mask = text_mask.flatten(1)

        fv1 = self.mhead1(vis_feat, text_feat)
        fc = self.mhead2(vis_feat, text_feat)
        fc = fc + vis_feat
        fv2 = self.mhead3(fc, vis_feat)
        vis_feat = fv1 + fv2

        # target regression token
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, B, 1)
        tgt_mask = torch.zeros((B, 1)).to(tgt_src.device).to(torch.bool)

        vl_src = torch.cat([tgt_src, text_feat, vis_feat], dim=0)
        vl_mask = torch.cat([tgt_mask, text_mask, vis_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, B, 1)

        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos) # [num_enc_layers, 1+L+HW/32*32, B, embedding_dim]

        if self.aux_loss:
            out_list = []
            for aux_pred in vg_hs:
                out_list.append(self.bbox_embed(aux_pred[0]).sigmoid())

            if self.training:
                return out_list

            return out_list[-1]

        vg_hs = vg_hs[-1][0] # [B, embedding_dim]
        pred_box = self.bbox_embed(vg_hs).sigmoid() # [B, 4]

        return pred_box