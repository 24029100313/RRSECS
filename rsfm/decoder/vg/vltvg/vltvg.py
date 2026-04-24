from torch import nn
from rsfm.utils import MLP
from .transformer import build_visual_encoder
from .decoder import build_vg_decoder
from .decoder_config import decoder_cfg



class VLTVGHead(nn.Module):
    '''
    trans_enc: Bool
        True: original implementation, using a transformer encoder to enhance basic features from backbone
        False: using features from backbone directly
    '''
    def __init__(self,
                 img_size=640,
                 embedding_dim=256,
                 in_channels=[96, 192, 384, 768],
                 num_classes=1,
                 dropout=0.1,
                 num_heads=8,
                 num_enc_layer=6,
                 aux_loss=False,
                 trans_enc=True, **kwargs):
        super(VLTVGHead, self).__init__()
        self.aux_loss = aux_loss
        self.trans_enc = trans_enc

        if trans_enc:
            self.backbone_transformer = build_visual_encoder(embedding_dim=embedding_dim,
                                                             dropout=dropout,
                                                             num_heads=num_heads,
                                                             dim_feedforward=2048,
                                                             num_enc_layer=num_enc_layer)

        self.input_proj = nn.Conv2d(in_channels[-1], embedding_dim, kernel_size=1)
        self.text_proj = nn.Linear(768, embedding_dim)  # 768 for bert-base

        # visual grounding
        decoder_cfg['extra_layer']['img2img_attn_args']['img_size'] = img_size
        self.trans_decoder = build_vg_decoder(decoder_cfg)
        self.bbox_embed = MLP(embedding_dim, embedding_dim, 4, 3)


    def forward(self, features, pos, text_feat, text_mask):
        vis_feat, vis_mask = features[-1].decompose()

        if self.trans_enc:
            vis_feat, vis_mask, pos_embed = self.backbone_transformer(self.input_proj(vis_feat), vis_mask, pos[-1])
        else:
            vis_feat = self.input_proj(vis_feat).flatten(2).permute(2, 0, 1)
            vis_mask = vis_mask.flatten(1)
            pos_embed = pos[-1].flatten(2).permute(2, 0, 1)

        text_feat = self.text_proj(text_feat) # [B, L, embedding_dim]
        text_feat = text_feat.permute(1, 0, 2) # [L, B, embedding_dim]
        text_mask = text_mask.flatten(1) # [B, L]

        # Discriminative feature encoding + Multi-stage reasoning
        hs = self.trans_decoder(vis_feat, vis_mask, pos_embed, text_feat, text_mask) # [6, B, 1, 256]
        outputs_coord = self.bbox_embed(hs).sigmoid() # [6, B, 1, 4]

        if self.aux_loss:
            out_list = []
            for aux_pred in outputs_coord:
                out_list.append(aux_pred.reshape(-1, 4))

            if self.training:
                return out_list

            return out_list[-1]

        out = outputs_coord[-1].reshape(-1, 4)

        return out