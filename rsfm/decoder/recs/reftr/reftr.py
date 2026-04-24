import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import build_vl_transformer, mlp_mapping, QueryEncoder, MaskHeadSmallConv, MHAttentionMap
from rsfm.utils import MLP, NestedTensor, build_position_encoding



class RefTRHead(nn.Module):
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 num_classes=2,
                 l_dim=768,
                 embedding_dim=256,
                 enc_layers=6,
                 dec_layers=6,
                 num_heads=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 aux_loss=False,
                 **kwargs):
        super(RefTRHead, self).__init__()
        # For RIS
        self.bbox_attention = MHAttentionMap(embedding_dim, embedding_dim, num_heads, dropout=0)
        self.mask_head = MaskHeadSmallConv(embedding_dim * 2 + num_heads, in_channels, embedding_dim, num_classes)

        # For VG
        self.vl_transformer = build_vl_transformer(hidden_dim=embedding_dim,
                                                   nheads=num_heads,
                                                   enc_layers=enc_layers,
                                                   dec_layers=dec_layers,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   num_feature_levels=1,
                                                   return_intermediate_dec=aux_loss)

        self.num_queries_per_phrase = 1
        self.hidden_dim = embedding_dim
        self.bbox_embed = MLP(embedding_dim, embedding_dim, 4, 3)

        self.input_proj = nn.Sequential(nn.Conv2d(in_channels[-1], embedding_dim, kernel_size=1),
                                        nn.GroupNorm(32, embedding_dim))

        self.map_sentence = mlp_mapping(l_dim, embedding_dim)

        self.map_phrase = mlp_mapping(l_dim, embedding_dim)

        self.query_encoder = QueryEncoder(
            num_queries_per_phrase=self.num_queries_per_phrase,
            hidden_dim=embedding_dim)

        self.aux_loss = aux_loss
        self.pos_embed = build_position_encoding()

        # initialization
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.xavier_uniform_(self.input_proj[0].weight, gain=1)
        nn.init.constant_(self.input_proj[0].bias, 0)


    def forward(self, features, x_mask, text_feat, text_mask):
        feat = self.input_proj(features[-1])
        mask = F.interpolate(x_mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
        pos = self.pos_embed(NestedTensor(feat, mask)).to(feat.dtype)

        srcs = [feat]
        masks = [mask]
        poses = [pos]

        phrase_pooled_feat = torch.mean(text_feat, dim=1)
        text_feat = self.map_sentence(text_feat)

        # Process phrase queries
        n_q = self.num_queries_per_phrase
        bsz = text_feat.size(0)
        n_ph = 1

        sentence_len = text_mask.to(torch.int32).sum(-1)
        mask_context = text_mask.view(bsz, n_ph, -1).logical_not().to(torch.bool)
        # Mask out [CLS] and [SEP]
        mask_context[:, :, 0] = True
        for i in range(bsz):
            mask_context[i, :, sentence_len[i] - 1] = True
        query_mask = torch.zeros((bsz, 1), device=text_mask.device).to(torch.bool)

        phrase_pooled_feat = self.map_phrase(phrase_pooled_feat).view(bsz, n_ph, -1)

        memory, memory_mask, memory_pos = self.vl_transformer.encode(img_srcs=srcs,
                                                                     img_masks=masks,
                                                                     img_pos_embeds=poses,
                                                                     lang_srcs=text_feat,
                                                                     lang_masks=text_mask)
        memory_lang = memory[:text_feat.size(1)]
        query, query_pos = self.query_encoder(lang_context_feat=memory_lang.transpose(0, 1),
                                              lang_query_feat=phrase_pooled_feat,
                                              mask_query_context=mask_context)

        hs = self.vl_transformer.decoder(
            tgt=query,
            memory=memory,
            tgt_key_padding_mask=query_mask,
            memory_key_padding_mask=memory_mask,
            query_pos=query_pos,
            pos=memory_pos,
        ).transpose(1, 2)

        num_l = hs.size(0)
        hs = hs.view(num_l, bsz, n_ph, n_q, -1)
        last_layer_hs = hs[-1]
        outputs_coord = self.bbox_embed(hs).sigmoid()

        # segmentation head
        pred_masks, _, _ = self.refer_segmentation(
            decoder_hs=last_layer_hs,
            memory_visual=memory[text_feat.size(1):].transpose(0, 1),
            img_src_proj=srcs[0],
            img_features=features,
            x_mask=x_mask
        )

        if self.aux_loss:
            out_list = []
            for aux_pred in outputs_coord:
                out_list.append(aux_pred.squeeze(1).squeeze(1))

            if self.training:
                out = {'pred_masks': pred_masks,
                       'pred_bboxs': out_list}

                return out

            return {'pred_masks': pred_masks,
                    'pred_bboxs': out_list[-1]}

        pred_box = outputs_coord[-1].squeeze(1).squeeze(1)
        out = {'pred_masks': pred_masks,
               'pred_bboxs': pred_box}

        return out


    def refer_segmentation(self, decoder_hs, memory_visual, img_src_proj, img_features, x_mask):
        img_src = img_features[-1]
        img_mask = F.interpolate(x_mask[None].float(), size=img_src.shape[-2:]).to(torch.bool)[0]

        bs, _, img_h, img_w = img_src.shape

        memory_visual = memory_visual.transpose(1, 2).view(bs, -1, img_h, img_w)
        assert memory_visual.shape == img_src_proj.shape
        img_src = torch.cat([img_src_proj, memory_visual], dim=1)

        # bbox_mask: [b, q, n, h, w]
        bbox_mask = self.bbox_attention(decoder_hs, memory_visual, mask=img_mask)
        seg_masks, res_feat = self.mask_head(img_src, bbox_mask, [img_features[2], img_features[1], img_features[0]])

        return seg_masks, bbox_mask, res_feat