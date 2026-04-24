import torch
from torch import nn
from rsfm.decoder.vg.transvg.vl_transformer import build_vl_transformer
from rsfm.utils import MLP
from einops import rearrange, repeat



class QRNetHead(nn.Module):
    def __init__(self,
                 img_size=640,
                 embedding_dim=256,
                 in_channels=[96, 192, 384, 768],
                 dropout=0.1,
                 num_heads=8,
                 num_enc_layer=6,
                 num_classes=1,
                 aux_loss=False,
                 **kwargs):
        super(QRNetHead, self).__init__()

        self.num_vis_token = int((img_size / 32) ** 2) + int((img_size / 64) ** 2)
        self.num_text_token = 20
        self.aux_loss = aux_loss

        num_total = self.num_vis_token + self.num_text_token + 1 # 400 + 20 + 1 = 421
        self.vl_pos_embed = nn.Embedding(num_total, embedding_dim)
        self.scale_embed = nn.Embedding(5, embedding_dim)
        self.reg_token = nn.Embedding(1, embedding_dim)

        self.vis_proj = nn.Linear(embedding_dim, embedding_dim)
        self.text_proj = nn.Linear(768, embedding_dim) # 768 for bert-base

        self.vl_transformer = build_vl_transformer(embedding_dim=embedding_dim,
                                                   dropout_rate=dropout,
                                                   num_heads=num_heads,
                                                   dim_feedforward=2048,
                                                   num_enc_layer=num_enc_layer,
                                                   return_intermediate=aux_loss)
        self.bbox_embed = MLP(embedding_dim, embedding_dim, 4, 3)


    def forward(self, features, pos, text_feat, text_mask):
        bs = text_feat.shape[0]

        text_feat = self.text_proj(text_feat) # [B, L, embedding_dim]
        text_feat = text_feat.permute(1, 0, 2) # [L, B, embedding_dim]
        text_mask = text_mask.flatten(1) # [B, L]

        x = [rearrange(feature.tensors, 'B C H W -> (H W) B C') for feature in features[-2:]]
        x_mask = [rearrange(feature.mask, 'B H W -> B (H W)') for feature in features[-2:]]

        vis_feat = torch.cat(x, dim=0)
        vis_mask = torch.cat(x_mask, dim=1)
        visu_scale=torch.cat([
            repeat(self.scale_embed.weight[-2], 'D -> L B D', B=bs, L=x[-2].shape[0]),
            repeat(self.scale_embed.weight[-1], 'D -> L B D', B=bs, L=x[-1].shape[0])
        ], dim=0)

        vis_feat = self.vis_proj(vis_feat + visu_scale)

        # target regression token
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1) # [1, B, embedding_dim]
        tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool) # [B, 1]

        vl_src = torch.cat([tgt_src, text_feat, vis_feat], dim=0) # [1+L+HW/32*32, B, embedding_dim]
        vl_mask = torch.cat([tgt_mask, text_mask, vis_mask], dim=1) # [B, 1+L+HW/32*32]
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1) # [1+L+HW/32*32, B, embedding_dim]

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