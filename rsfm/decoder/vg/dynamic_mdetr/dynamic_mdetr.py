import torch
from torch import nn
from .vl_encoder import build_vl_encoder
from .vl_transformer import build_vl_transformer
from rsfm.utils import MLP
from torch.nn.functional import grid_sample



class DynamicMDETRHead(nn.Module):
    '''
    from 'Dynamic MDETR: A Dynamic Multimodal Transformer Decoder for Visual Grounding, TPAMI 2023'
    '''
    def __init__(self,
                 img_size=640,
                 embedding_dim=256,
                 in_channels=[96, 192, 384, 768],
                 dropout=0.1,
                 num_heads=8,
                 num_classes=1,
                 aux_loss=False,
                 **kwargs):
        super(DynamicMDETRHead, self).__init__()

        self.num_vis_token = int((img_size / 32) ** 2) # (640 / 32)^2 = 400
        self.num_text_token = 20

        self.uniform_learnable = True
        self.different_transformer = True
        self.different_transformer = True

        self.aux_loss = aux_loss

        num_total = self.num_vis_token + self.num_text_token # 400 + 20 = 420
        self.vl_pos_embed = nn.Embedding(num_total, embedding_dim)
        self.vl_encoder = build_vl_encoder(vl_hidden_dim=embedding_dim,
                                           vl_dropout=dropout,
                                           vl_nheads=num_heads,
                                           vl_dim_feedforward=2048,
                                           vl_fusion_enc_layers=3)

        self.vis_proj = nn.Linear(in_channels[-1], embedding_dim)
        self.text_proj = nn.Linear(768, embedding_dim) # 768 for bert-base

        self.visual_feature_map_h = int(img_size / 32)
        self.visual_feature_map_w = int(img_size / 32)
        self.in_points = 36
        self.stages = 3

        self.offset_generators = nn.ModuleList([nn.Linear(embedding_dim, self.in_points * 2)
                                                for _ in range(self.stages)])
        self.update_sampling_queries = nn.ModuleList([MLP(2 * embedding_dim, embedding_dim, embedding_dim, 2)
                                                     for _ in range(self.stages)])

        self.init_reference_point = nn.Embedding(1, 2)
        self.init_sampling_feature = nn.Embedding(1, embedding_dim)

        self.init_weights()

        if self.different_transformer:
            self.vl_transformer = nn.ModuleList([
                build_vl_transformer(vl_hidden_dim=embedding_dim,
                                     vl_dropout=dropout,
                                     vl_nheads=num_heads,
                                     vl_dim_feedforward=2048,
                                     vl_enc_layers=1,
                                     vl_dec_layers=1)
                for _ in range(self.stages)])
        else:
            self.vl_transformer = build_vl_transformer(vl_hidden_dim=embedding_dim,
                                                       vl_dropout=dropout,
                                                       vl_nheads=num_heads,
                                                       vl_dim_feedforward=2048,
                                                       vl_enc_layers=1,
                                                       vl_dec_layers=1)

        self.bbox_embed = MLP(embedding_dim, embedding_dim, 4, 3)


    def init_weights(self):
        nn.init.constant_(self.init_reference_point.weight[:, 0], 0.5)
        nn.init.constant_(self.init_reference_point.weight[:, 1], 0.5)
        self.init_reference_point.weight.requires_grad = False

        for i in range(self.stages):
            nn.init.zeros_(self.offset_generators[i].weight)
            nn.init.uniform_(self.offset_generators[i].bias, -0.5, 0.5)
        if not self.uniform_learnable:
            self.offset_generators[0].weight.requires_grad = False
            self.offset_generators[0].bias.requires_grad = False


    def feautures_sampling(self, sampling_query, reference_point, feature_map, pos, stage):
        bs, channel = sampling_query.shape
        xy_offsets = self.offset_generators[stage](sampling_query).reshape(bs, self.in_points, 2)
        sampled_points = (xy_offsets.permute(1, 0, 2) + reference_point).permute(1, 0, 2)  # (bs, in_points, 2)
        feature_map = feature_map.reshape(bs, channel, self.visual_feature_map_h, self.visual_feature_map_w) # (bs, channel, h, w)
        pos = pos.reshape(bs, channel, self.visual_feature_map_h, self.visual_feature_map_w) # (bs, channel, h, w)

        sampled_points = (2 * sampled_points) - 1

        sampled_features = grid_sample(feature_map, sampled_points.unsqueeze(2), mode='bilinear', padding_mode='border',
                                       align_corners=False).squeeze(-1)  # (bs, channel, in_points)
        pe = grid_sample(pos, sampled_points.unsqueeze(2), mode='bilinear', padding_mode='border', align_corners=False).squeeze(-1) # (bs, channel, in_points)

        return sampled_features, pe


    def forward(self, features, pos, text_src, text_mask):
        vis_src, vis_mask = features[-1].decompose()
        B = vis_src.shape[0]

        vis_mask, vis_src = vis_mask.flatten(1), vis_src.flatten(2).permute(2, 0, 1)
        vis_src = self.vis_proj(vis_src) # [HW/32*32, B, embedding_dim]

        text_src = self.text_proj(text_src).permute(1, 0, 2) # [L, B, embedding_dim]
        text_mask = text_mask.flatten(1) # [B, L]

        vl_src = torch.cat([vis_src, text_src], dim=0)
        vl_mask = torch.cat([vis_mask, text_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, B, 1)

        if self.vl_encoder is not None:
            vl_feat = self.vl_encoder(vl_src, vl_mask, vl_pos)  # (L+N)xBxC
        else:
            vl_feat = vl_src

        vis_feat = vl_feat[:self.num_vis_token]  # (H*W, B, channel)
        language_feat = vl_feat[self.num_vis_token:]  # (max_len, B, channel)
        v_pos = vl_pos[:self.num_vis_token]
        l_pos = vl_pos[self.num_vis_token:]

        sampling_query = self.init_sampling_feature.weight.repeat(B, 1)
        reference_point = self.init_reference_point.weight.repeat(B, 1)

        out_list = []
        for i in range(0, self.stages):
            # 2D adaptive sampling
            sampled_features, pe = self.feautures_sampling(sampling_query, reference_point, vis_feat.permute(1, 2, 0), v_pos.permute(1, 2, 0), i)

            # Text guided decoding with one-layer transformer encoder-decoder
            if self.different_transformer:
                vg_hs = self.vl_transformer[i](sampled_features, None, language_feat, pe, text_mask, l_pos)[0]
            else:
                vg_hs = self.vl_transformer(sampled_features, None, language_feat, pe, text_mask, l_pos)[0]

            # Prediction Head
            language_feat = vg_hs[0]

            text_select = (1 - text_mask * 1.0).unsqueeze(-1)  # (bs, max_len, 1)
            text_select_num = text_select.sum(dim=1)  # (bs, 1)

            # new language queries
            vg_hs = (text_select * vg_hs[0].permute(1,0,2)).sum(dim=1) / text_select_num  # (bs, channel)
            pred_box = self.bbox_embed(vg_hs).sigmoid()
            out_list.append(pred_box)

            # Update reference point and sampling query
            reference_point = pred_box[:, :2]
            sampling_query = self.update_sampling_queries[i](torch.cat((vg_hs, sampling_query), dim=1))

        if self.aux_loss:
            if self.training:
                return out_list
            return out_list[-1]

        return out_list[-1]