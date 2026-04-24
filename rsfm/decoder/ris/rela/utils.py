import torch
from torch import nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from rsfm.utils import build_position_encoding, MLP, NestedTensor
from ...seg.mask2former.transformer_decoder.utils import SelfAttentionLayer, CrossAttentionLayer, FFNLayer



class MultiScaleMaskedReferringDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 mask_classification=True,
                 hidden_dim=256,
                 num_queries=100,
                 nheads=8,
                 dim_feedforward=2048,
                 dec_layers=10,
                 pre_norm=False,
                 mask_dim=256,
                 enforce_input_project=False,
                 rla_weight=0.1):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        self.pe_layer = build_position_encoding(hidden_dim, 'sine')

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.RLA_vision = nn.ModuleList()
        self.RIA_layers = nn.ModuleList()
        self.rla_weight = rla_weight
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.RLA_vision.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.RIA_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.lang_proj = nn.Linear(768, hidden_dim, False)
        self.lang_weight = nn.parameter.Parameter(data=torch.as_tensor(0.))
        self.RLA_lang_att = CrossAttentionLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dropout=0.0,
            normalize_before=pre_norm,
        )

        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        nn.init.zeros_(self.lang_proj.weight)

        # output FFNs
        if self.mask_classification:
            self.minimap_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.nt_embed = MLP(hidden_dim, hidden_dim, 2, 2) # nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)


    def forward(self, x, mask_features, lang_feat, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(
                NestedTensor(x[i],
                             torch.zeros((x[i].size(0), x[i].size(2), x[i].size(3)), device=x[i].device, dtype=torch.bool))
            ).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_minimap, outputs_mask, attn_mask, tgt_mask, nt_label = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_minimap)
        predictions_mask.append(outputs_mask)

        # ReLA is applied multiple times for performance
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross-attention of regions and vision features
            output = self.RIA_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )

            if i == 0:
                # For the first layer, apply full RLA
                # Later, apply region attention only for memory saving
                lang_feat_att = lang_feat.permute(0,2,1)
                lang_feat_att = self.lang_proj(lang_feat_att)
                lang_feat_att = self.RLA_lang_att(output, lang_feat_att.permute(1,0,2)) * F.sigmoid(self.lang_weight)
                output = output + lang_feat_att * self.rla_weight

            # RLA vision attention
            # self attention itself has a skip connection
            output = self.RLA_vision[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # Postprocessing
            output = self.transformer_ffn_layers[i](output)

            outputs_minimap, outputs_mask, attn_mask, tgt_mask, nt_label = self.forward_prediction_heads(
                output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])

            # Predictions of all passes are recorded, but only the last output is used in this code
            predictions_class.append(outputs_minimap)
            predictions_mask.append(outputs_mask)

        out = {
            'pred_logits': predictions_class[-1], # [B, Q, num_classes + 1]
            'pred_masks': tgt_mask, # [ B, num_classes + 1, H/4, W/4]
            'all_masks': outputs_mask,
            'nt_label': nt_label # [B, 2]
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        region_features = self.decoder_norm(output)
        region_features = region_features.transpose(0, 1)

        region_embed = self.mask_embed(region_features)
        # F_region * F_mask -> all masks
        # Note: here we used region feature after RLA for mask prediction
        #       this can greatly enhance the training stability
        all_mask = torch.einsum("bqc,bchw->bqhw", region_embed, mask_features)

        # F_minimap * F_region * F_mask -> target mask
        outputs_minimap = self.minimap_embed(region_features)
        tgt_embed = torch.einsum("bqa,bqc->bac", outputs_minimap, region_embed)
        tgt_mask = torch.einsum("bac,bchw->bahw", tgt_embed, mask_features)

        # F_region -> NT_label
        nt_label = self.nt_embed(region_features)
        nt_label = nt_label.mean(dim=1) # Global average pooling

        attn_mask = F.interpolate(all_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_minimap, all_mask, attn_mask, tgt_mask, nt_label