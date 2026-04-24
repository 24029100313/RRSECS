import torch
import torch.nn.functional as F
from torch import nn
import math
from .deformable_transformer import build_deforamble_transformer
from rsfm.utils import MLP, get_clones, inverse_sigmoid, NestedTensor, build_position_encoding




# class DeformableDETRHead(nn.Module):
#     """ This is the Deformable DETR module that performs object detection """
#     def __init__(self,
#                  embedding_dim=256,
#                  in_channels=[96, 192, 384, 768],
#                  num_classes=10, # 1 for VG
#                  num_queries=300, # 1 for VG
#                  num_enc_layer=6,
#                  num_dec_layer=6,
#                  dropout=0.1,
#                  num_heads=8,
#                  num_feature_levels=4,
#                  aux_loss=True,
#                  with_box_refine=True,
#                  two_stage=True, # False for VG
#                  **kwargs):
#         super().__init__()
#         self.transformer = build_deforamble_transformer(embedding_dim=embedding_dim,
#                                                         num_heads=num_heads,
#                                                         enc_layers=num_enc_layer,
#                                                         dec_layers=num_dec_layer,
#                                                         dim_feedforward=1024,
#                                                         dropout=dropout,
#                                                         num_feature_levels=num_feature_levels,
#                                                         two_stage=two_stage,
#                                                         num_queries=num_queries)
#         self.pos_embed = build_position_encoding()
#
#         self.num_queries = num_queries
#         self.class_embed = nn.Linear(embedding_dim, num_classes)
#         self.bbox_embed = MLP(embedding_dim, embedding_dim, 4, 3)
#         self.num_feature_levels = num_feature_levels
#
#         if not two_stage:
#             self.query_embed = nn.Embedding(num_queries, embedding_dim*2)
#
#         if num_feature_levels > 1:
#             num_backbone_outs = len(in_channels[(1 - num_feature_levels):])
#             input_proj_list = []
#             for _ in range(num_backbone_outs):
#                 in_channel = in_channels[(1 - num_feature_levels):][_]
#                 input_proj_list.append(nn.Sequential(
#                     nn.Conv2d(in_channel, embedding_dim, kernel_size=1),
#                     nn.GroupNorm(32, embedding_dim),
#                 ))
#             for _ in range(num_feature_levels - num_backbone_outs):
#                 input_proj_list.append(nn.Sequential(
#                     nn.Conv2d(in_channel, embedding_dim, kernel_size=3, stride=2, padding=1),
#                     nn.GroupNorm(32, embedding_dim),
#                 ))
#                 in_channel = embedding_dim
#             self.input_proj = nn.ModuleList(input_proj_list)
#         else:
#             self.input_proj = nn.ModuleList([
#                 nn.Sequential(
#                     nn.Conv2d(in_channels[-1], embedding_dim, kernel_size=1),
#                     nn.GroupNorm(32, embedding_dim),
#                 )])
#
#         self.aux_loss = aux_loss
#         self.with_box_refine = with_box_refine
#         self.two_stage = two_stage
#
#         prior_prob = 0.01
#         bias_value = -math.log((1 - prior_prob) / prior_prob)
#         self.class_embed.bias.data = torch.ones(num_classes) * bias_value
#         nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
#         nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
#         for proj in self.input_proj:
#             nn.init.xavier_uniform_(proj[0].weight, gain=1)
#             nn.init.constant_(proj[0].bias, 0)
#
#         # if two-stage, the last class_embed and bbox_embed is for region proposal generation
#         num_pred = (self.transformer.decoder.num_layers + 1) if two_stage else self.transformer.decoder.num_layers
#         if with_box_refine:
#             self.class_embed = get_clones(self.class_embed, num_pred)
#             self.bbox_embed = get_clones(self.bbox_embed, num_pred)
#             nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
#             # hack implementation for iterative bounding box refinement
#             self.transformer.decoder.bbox_embed = self.bbox_embed
#         else:
#             nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
#             self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
#             self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
#             self.transformer.decoder.bbox_embed = None
#         if two_stage:
#             # hack implementation for two-stage
#             self.transformer.decoder.class_embed = self.class_embed
#             for box_embed in self.bbox_embed:
#                 nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
#
#
#     def forward(self, features, pos):
#         srcs = []
#         masks = []
#         for l, feat in enumerate(features[(1 - self.num_feature_levels):]):
#             src, mask = feat.decompose()
#             srcs.append(self.input_proj[l](src))
#             masks.append(mask)
#             assert mask is not None
#         if self.num_feature_levels > len(srcs):
#             _len_srcs = len(srcs)
#             for l in range(_len_srcs, self.num_feature_levels):
#                 if l == _len_srcs:
#                     src = self.input_proj[l](features[-1].tensors)
#                 else:
#                     src = self.input_proj[l](srcs[-1])
#                 m = features[-1].mask
#                 mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
#                 pos_l = self.pos_embed(NestedTensor(src, mask)).to(src.dtype)
#                 srcs.append(src)
#                 masks.append(mask)
#                 pos.append(pos_l)
#         pos = pos[(len(pos) - self.num_feature_levels):]
#
#         query_embeds = None
#         if not self.two_stage:
#             query_embeds = self.query_embed.weight
#         hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)
#
#         outputs_classes = []
#         outputs_coords = []
#         for lvl in range(hs.shape[0]):
#             if lvl == 0:
#                 reference = init_reference
#             else:
#                 reference = inter_references[lvl - 1]
#             reference = inverse_sigmoid(reference)
#             outputs_class = self.class_embed[lvl](hs[lvl])
#             tmp = self.bbox_embed[lvl](hs[lvl])
#             if reference.shape[-1] == 4:
#                 tmp += reference
#             else:
#                 assert reference.shape[-1] == 2
#                 tmp[..., :2] += reference
#             outputs_coord = tmp.sigmoid()
#             outputs_classes.append(outputs_class)
#             outputs_coords.append(outputs_coord)
#         outputs_class = torch.stack(outputs_classes)
#         outputs_coord = torch.stack(outputs_coords)
#
#         out = {'pred_logits': outputs_class[-1],
#                'pred_boxes': outputs_coord[-1]}
#         if self.aux_loss:
#             out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
#
#         if self.two_stage:
#             enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
#             out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
#         return out
#
#     @torch.jit.unused
#     def _set_aux_loss(self, outputs_class, outputs_coord):
#         return [{'pred_logits': a, 'pred_boxes': b}
#                 for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]



# class DeformableDETRHead(nn.Module):
#     def __init__(self,
#                  embedding_dim=256,
#                  in_channels=[96, 192, 384, 768],
#                  num_classes=1, # 1 for VG
#                  num_queries=10, # 1 for VG
#                  num_enc_layer=6,
#                  num_dec_layer=6,
#                  dropout=0.1,
#                  num_heads=8,
#                  num_feature_levels=4,
#                  aux_loss=True,
#                  with_box_refine=True,
#                  two_stage=False, # False for VG
#                  **kwargs):
#         super().__init__()
#         self.transformer = build_deforamble_transformer(embedding_dim=embedding_dim,
#                                                         num_heads=num_heads,
#                                                         enc_layers=num_enc_layer,
#                                                         dec_layers=num_dec_layer,
#                                                         dim_feedforward=1024,
#                                                         dropout=dropout,
#                                                         num_feature_levels=num_feature_levels,
#                                                         two_stage=two_stage,
#                                                         num_queries=num_queries)
#         self.pos_embed = build_position_encoding()
#
#         self.num_queries = num_queries
#         self.class_embed = nn.Linear(embedding_dim, num_classes)
#         self.bbox_embed = MLP(embedding_dim, embedding_dim, 4, 3)
#         self.num_feature_levels = num_feature_levels
#
#         if not two_stage:
#             self.query_embed = nn.Embedding(num_queries, embedding_dim*2)
#
#         if num_feature_levels > 1:
#             num_backbone_outs = len(in_channels[(1 - num_feature_levels):])
#             input_proj_list = []
#             for _ in range(num_backbone_outs):
#                 in_channel = in_channels[(1 - num_feature_levels):][_]
#                 input_proj_list.append(nn.Sequential(
#                     nn.Conv2d(in_channel, embedding_dim, kernel_size=1),
#                     nn.GroupNorm(32, embedding_dim),
#                 ))
#             for _ in range(num_feature_levels - num_backbone_outs):
#                 input_proj_list.append(nn.Sequential(
#                     nn.Conv2d(in_channel, embedding_dim, kernel_size=3, stride=2, padding=1),
#                     nn.GroupNorm(32, embedding_dim),
#                 ))
#                 in_channel = embedding_dim
#             self.input_proj = nn.ModuleList(input_proj_list)
#         else:
#             self.input_proj = nn.ModuleList([
#                 nn.Sequential(
#                     nn.Conv2d(in_channels[-1], embedding_dim, kernel_size=1),
#                     nn.GroupNorm(32, embedding_dim),
#                 )])
#
#         self.aux_loss = aux_loss
#         self.with_box_refine = with_box_refine
#         self.two_stage = two_stage
#
#         prior_prob = 0.01
#         bias_value = -math.log((1 - prior_prob) / prior_prob)
#         self.class_embed.bias.data = torch.ones(num_classes) * bias_value
#         nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
#         nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
#         for proj in self.input_proj:
#             nn.init.xavier_uniform_(proj[0].weight, gain=1)
#             nn.init.constant_(proj[0].bias, 0)
#
#         # if two-stage, the last class_embed and bbox_embed is for region proposal generation
#         num_pred = (self.transformer.decoder.num_layers + 1) if two_stage else self.transformer.decoder.num_layers
#         if with_box_refine:
#             self.class_embed = get_clones(self.class_embed, num_pred)
#             self.bbox_embed = get_clones(self.bbox_embed, num_pred)
#             nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
#             # hack implementation for iterative bounding box refinement
#             self.transformer.decoder.bbox_embed = self.bbox_embed
#         else:
#             nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
#             self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
#             self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
#             self.transformer.decoder.bbox_embed = None
#         if two_stage:
#             # hack implementation for two-stage
#             self.transformer.decoder.class_embed = self.class_embed
#             for box_embed in self.bbox_embed:
#                 nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
#
#
#     def forward(self, features, pos):
#         srcs = []
#         masks = []
#         for l, feat in enumerate(features[(1 - self.num_feature_levels):]):
#             src, mask = feat.decompose()
#             srcs.append(self.input_proj[l](src))
#             masks.append(mask)
#             assert mask is not None
#         if self.num_feature_levels > len(srcs):
#             _len_srcs = len(srcs)
#             for l in range(_len_srcs, self.num_feature_levels):
#                 if l == _len_srcs:
#                     src = self.input_proj[l](features[-1].tensors)
#                 else:
#                     src = self.input_proj[l](srcs[-1])
#                 m = features[-1].mask
#                 mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
#                 pos_l = self.pos_embed(NestedTensor(src, mask)).to(src.dtype)
#                 srcs.append(src)
#                 masks.append(mask)
#                 pos.append(pos_l)
#         pos = pos[(len(pos) - self.num_feature_levels):]
#
#         query_embeds = None
#         if not self.two_stage:
#             query_embeds = self.query_embed.weight
#         hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)
#
#         outputs_classes = []
#         outputs_coords = []
#         for lvl in range(hs.shape[0]):
#             if lvl == 0:
#                 reference = init_reference
#             else:
#                 reference = inter_references[lvl - 1]
#             reference = inverse_sigmoid(reference)
#             outputs_class = self.class_embed[lvl](hs[lvl])
#             tmp = self.bbox_embed[lvl](hs[lvl])
#             if reference.shape[-1] == 4:
#                 tmp += reference
#             else:
#                 assert reference.shape[-1] == 2
#                 tmp[..., :2] += reference
#             outputs_coord = tmp.sigmoid()
#             outputs_classes.append(outputs_class)
#             outputs_coords.append(outputs_coord)
#         outputs_class = torch.stack(outputs_classes) # num_layers, B, Q, num_classes
#         outputs_coord = torch.stack(outputs_coords) # num_layers, B, Q, 4
#
#
#         outputs_class, outputs_coord = outputs_class.unsqueeze(2), outputs_coord.unsqueeze(2)
#         if self.aux_loss:
#             out_list = []
#             for out_cls, out_coord in zip(outputs_class, outputs_coord):
#                 out_list.append(
#                     {'pred_logits': out_cls,  # [B, 1, Q, 1]
#                      'pred_boxes': out_coord  # [B, 1, Q, 4]
#                      })
#
#             if self.training:
#                 return out_list
#
#             return self.out_online_eval(out_list[-1])
#
#         out = {}
#         out['pred_logits'] = outputs_class[-1]
#         out['pred_boxes'] = outputs_coord[-1]
#
#         if self.training:
#             return out
#
#         return self.out_online_eval(out)
#
#     def out_online_eval(self, outputs):
#         assert outputs["pred_boxes"].shape[0] == 1
#
#         pred_logits = outputs["pred_logits"][0]
#         pred_bbox = outputs["pred_boxes"][0]
#         pred_score = pred_logits.sigmoid()  # [t, q, k]
#         pred_score = pred_score.squeeze(0)  # [q, k]
#         max_score, _ = pred_score.max(-1)  # [q,]
#         _, max_ind = max_score.max(-1)  # [1,] # which query
#         pred_bbox = pred_bbox[0, max_ind].unsqueeze(0)  # [xc, yc, w_b, h_b]
#
#         return pred_bbox



class DeformableDETRHead(nn.Module):
    def __init__(self,
                 embedding_dim=256,
                 in_channels=[96, 192, 384, 768],
                 num_classes=1, # 1 for VG
                 num_queries=10, # 1 for VG
                 num_enc_layer=6,
                 num_dec_layer=6,
                 dropout=0.1,
                 num_heads=8,
                 num_feature_levels=4,
                 aux_loss=True,
                 with_box_refine=True,
                 **kwargs):
        super().__init__()

        self.transformer = build_deforamble_transformer(embedding_dim=embedding_dim,
                                                        num_heads=num_heads,
                                                        enc_layers=num_enc_layer,
                                                        dec_layers=num_dec_layer,
                                                        dim_feedforward=1024,
                                                        dropout=dropout,
                                                        num_feature_levels=num_feature_levels,
                                                        two_stage=False,
                                                        num_queries=num_queries)

        self.num_queries = num_queries
        self.class_embed = nn.Linear(embedding_dim, num_classes)
        self.bbox_embed = MLP(embedding_dim, embedding_dim, 4, 3)
        self.num_feature_levels = num_feature_levels

        self.query_embed = nn.Embedding(num_queries, embedding_dim*2)

        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        num_pred = self.transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = get_clones(self.class_embed, num_pred)
            self.bbox_embed = get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None


    def forward(self, features, pos):
        srcs, masks = [], []
        for feat in features:
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)

        query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes) # num_layers, B, Q, num_classes
        outputs_coord = torch.stack(outputs_coords) # num_layers, B, Q, 4

        outputs_class, outputs_coord = outputs_class.unsqueeze(2), outputs_coord.unsqueeze(2)
        if self.aux_loss:
            out_list = []
            for out_cls, out_coord in zip(outputs_class, outputs_coord):
                out_list.append(
                    {'pred_logits': out_cls,  # [B, 1, Q, 1]
                     'pred_boxes': out_coord  # [B, 1, Q, 4]
                     })

            if self.training:
                return out_list

            return self.out_online_eval(out_list[-1])

        out = {}
        out['pred_logits'] = outputs_class[-1]
        out['pred_boxes'] = outputs_coord[-1]

        if self.training:
            return out

        return self.out_online_eval(out)

    def out_online_eval(self, outputs):
        assert outputs["pred_boxes"].shape[0] == 1

        pred_logits = outputs["pred_logits"][0]
        pred_bbox = outputs["pred_boxes"][0]
        pred_score = pred_logits.sigmoid()  # [t, q, k]
        pred_score = pred_score.squeeze(0)  # [q, k]
        max_score, _ = pred_score.max(-1)  # [q,]
        _, max_ind = max_score.max(-1)  # [1,] # which query
        pred_bbox = pred_bbox[0, max_ind].unsqueeze(0)  # [xc, yc, w_b, h_b]

        return pred_bbox



# class DeformableDETRHead(nn.Module):
#     def __init__(self,
#                  embedding_dim=256,
#                  in_channels=[96, 192, 384, 768],
#                  num_classes=1, # 1 for VG
#                  num_queries=10, # 1 for VG
#                  num_enc_layer=6,
#                  num_dec_layer=6,
#                  dropout=0.1,
#                  num_heads=8,
#                  num_feature_levels=4,
#                  aux_loss=True,
#                  with_box_refine=True,
#                  **kwargs):
#         super().__init__()
#
#         self.transformer = build_deforamble_transformer(embedding_dim=embedding_dim,
#                                                         num_heads=num_heads,
#                                                         enc_layers=num_enc_layer,
#                                                         dec_layers=num_dec_layer,
#                                                         dim_feedforward=1024,
#                                                         dropout=dropout,
#                                                         num_feature_levels=num_feature_levels,
#                                                         two_stage=False,
#                                                         num_queries=num_queries)
#
#         self.num_queries = num_queries
#         self.class_embed = nn.Linear(embedding_dim, num_classes)
#         self.bbox_embed = MLP(embedding_dim, embedding_dim, 4, 3)
#         self.num_feature_levels = num_feature_levels
#
#         self.query_embed = nn.Embedding(num_queries, embedding_dim*2)
#
#         self.aux_loss = aux_loss
#         self.with_box_refine = with_box_refine
#
#         prior_prob = 0.01
#         bias_value = -math.log((1 - prior_prob) / prior_prob)
#         self.class_embed.bias.data = torch.ones(num_classes) * bias_value
#         nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
#         nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
#
#         num_pred = self.transformer.decoder.num_layers
#         if with_box_refine:
#             self.class_embed = get_clones(self.class_embed, num_pred)
#             self.bbox_embed = get_clones(self.bbox_embed, num_pred)
#             nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
#             self.transformer.decoder.bbox_embed = self.bbox_embed
#         else:
#             nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
#             self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
#             self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
#             self.transformer.decoder.bbox_embed = None
#
#
#     def forward(self, features, pos):
#         srcs, masks = [], []
#         for feat in features:
#             src, mask = feat.decompose()
#             srcs.append(src)
#             masks.append(mask)
#
#         query_embeds = self.query_embed.weight
#         hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)
#
#         outputs_classes = []
#         outputs_coords = []
#         for lvl in range(hs.shape[0]):
#             if lvl == 0:
#                 reference = init_reference
#             else:
#                 reference = inter_references[lvl - 1]
#             reference = inverse_sigmoid(reference)
#             outputs_class = self.class_embed[lvl](hs[lvl])
#             tmp = self.bbox_embed[lvl](hs[lvl])
#             if reference.shape[-1] == 4:
#                 tmp += reference
#             else:
#                 assert reference.shape[-1] == 2
#                 tmp[..., :2] += reference
#             outputs_coord = tmp.sigmoid()
#             outputs_classes.append(outputs_class)
#             outputs_coords.append(outputs_coord)
#         outputs_class = torch.stack(outputs_classes) # num_layers, B, Q, num_classes
#         outputs_coord = torch.stack(outputs_coords) # num_layers, B, Q, 4
#
#         out = {'pred_logits': outputs_class[-1],
#                'pred_boxes': outputs_coord[-1]}
#         if self.aux_loss:
#             out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
#
#         if self.training:
#             return out
#
#         return self.out_online_eval(out)
#
#     @torch.jit.unused
#     def _set_aux_loss(self, outputs_class, outputs_coord):
#         return [{'pred_logits': a, 'pred_boxes': b}
#                 for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
#
#     def out_online_eval(self, outputs):
#         assert outputs["pred_boxes"].shape[0] == 1
#
#         pred_logits = outputs["pred_logits"][0]
#         pred_bbox = outputs["pred_boxes"][0]
#         pred_score = pred_logits.sigmoid()  # [q, k]
#         max_score, _ = pred_score.max(-1)  # [q,]
#         _, max_ind = max_score.max(-1)  # [1,] # which query
#         pred_bbox = pred_bbox[max_ind].unsqueeze(0)  # [xc, yc, w_b, h_b]
#
#         return pred_bbox