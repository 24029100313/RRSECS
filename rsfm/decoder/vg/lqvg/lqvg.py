import torch.nn.functional as F
from einops import rearrange, repeat
from rsfm.utils import MLP, get_clones, inverse_sigmoid, build_position_encoding
from .deformable_transformer import build_deforamble_transformer
from .module import *


class LQVGHead(nn.Module):
    def __init__(self,
                 img_size=640,
                 embedding_dim=256,
                 in_channels=[96, 192, 384, 768],
                 num_classes=1,
                 num_queries=1,
                 num_feature_levels=4,
                 with_box_refine=True,
                 aux_loss=False,
                 trans_enc=False,
                 **kwargs):
        super(LQVGHead, self).__init__()
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss

        self.pos_embed = build_position_encoding()

        self.class_embed = nn.Linear(embedding_dim, 1)
        self.bbox_embed = MLP(embedding_dim, embedding_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, embedding_dim)

        if num_feature_levels > 1:
            num_backbone_outs = len(in_channels[-3:])
            vis_proj_list = []
            for _ in range(num_backbone_outs):
                in_channel = in_channels[-3:][_]
                vis_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channel, embedding_dim, kernel_size=1),
                    nn.GroupNorm(32, embedding_dim)
                ))
            for _ in range(num_feature_levels - num_backbone_outs):  # downsample 2x
                vis_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channel, embedding_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, embedding_dim),
                ))
                in_channel = embedding_dim
            self.vis_proj = nn.ModuleList(vis_proj_list)
        else:
            self.vis_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels[-3:][0], embedding_dim, kernel_size=1),
                    nn.GroupNorm(32, embedding_dim),
                )
            ])

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(1) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.vis_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = build_deforamble_transformer(embedding_dim=embedding_dim,
                                                        dim_feedforward=2048,
                                                        return_intermediate_dec=True,
                                                        num_feature_levels=num_feature_levels,
                                                        num_queries=num_queries)

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

        self.text_proj = FeatureResizer(768, embedding_dim, 0.1)

        self.lvi_module = VisionLanguageFusionModule(d_model=embedding_dim, nhead=8)
        self.vli_module = VisionLanguageFusionModule(d_model=embedding_dim, nhead=8)

        self.text_pos = PositionEmbeddingSine1D(embedding_dim, normalize=True)
        self.poolout_module = RobertaPoolout(d_model=embedding_dim)


    def forward(self, features, pos, text_feat, text_mask):
        b = text_feat.shape[0]
        t = pos[0].shape[0] // b

        text_feat = self.text_proj(text_feat)  # [B, L, embedding_dim]
        text_features = NestedTensor(text_feat, text_mask)

        # prepare vision and text features for transformer
        srcs = []
        masks = []
        poses = []

        text_pos = self.text_pos(text_features).permute(2, 0, 1)  # [length, batch_size, c]
        text_word_features, text_word_masks = text_features.decompose()

        text_word_features = text_word_features.permute(1, 0, 2)  # [length, batch_size, c]
        text_word_initial_features = text_word_features

        # Follow Deformable-DETR, we use the last three stages outputs from backbone
        for l, (feat, pos_l) in enumerate(zip(features[-3:], pos[-3:])):
            src, mask = feat.decompose()
            src_proj_l = self.vis_proj[l](src)
            n, c, h, w = src_proj_l.shape

            # vision language early-fusion
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> (t h w) b c', b=b, t=t)
            mask = rearrange(mask, '(b t) h w -> b (t h w)', b=b, t=t)
            pos_l = rearrange(pos_l, '(b t) c h w -> (t h w) b c', b=b, t=t)
            text_word_features = self.lvi_module(tgt=text_word_features,
                                                 memory=src_proj_l,
                                                 memory_key_padding_mask=mask,
                                                 pos=pos_l,
                                                 query_pos=None)

            src_proj_l = self.vli_module(tgt=src_proj_l,
                                         memory=text_word_initial_features,
                                         memory_key_padding_mask=text_word_masks,
                                         pos=text_pos,
                                         query_pos=None)

            src_proj_l = rearrange(src_proj_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            mask = rearrange(mask, 'b (t h w) -> (b t) h w', t=t, h=h, w=w)
            pos_l = rearrange(pos_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)

            assert mask is not None

        if self.num_feature_levels > (len(features) - 1):
            _len_srcs = len(features) - 1  # fpn level
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.vis_proj[l](features[-1].tensors)
                else:
                    src = self.vis_proj[l](srcs[-1])

                m = features[-1].mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.pos_embed(NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                # vision language early-fusion
                src = rearrange(src, '(b t) c h w -> (t h w) b c', b=b, t=t)
                mask = rearrange(mask, '(b t) h w -> b (t h w)', b=b, t=t)
                pos_l = rearrange(pos_l, '(b t) c h w -> (t h w) b c', b=b, t=t)

                text_word_features = self.lvi_module(tgt=text_word_features,
                                                     memory=src,
                                                     memory_key_padding_mask=mask,
                                                     pos=pos_l,
                                                     query_pos=None)
                src = self.vli_module(tgt=src,
                                      memory=text_word_initial_features,
                                      memory_key_padding_mask=text_word_masks,
                                      pos=text_pos,
                                      query_pos=None)

                src = rearrange(src, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
                mask = rearrange(mask, 'b (t h w) -> (b t) h w', t=t, h=h, w=w)
                pos_l = rearrange(pos_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)

                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        text_word_features = rearrange(text_word_features, 'l b c -> b l c')
        text_sentence_features = self.poolout_module(text_word_features)

        # Transformer
        query_embeds = self.query_embed.weight  # [num_queries, c]
        text_embed = repeat(text_sentence_features, 'b c -> b t q c', t=t, q=self.num_queries)
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
            self.transformer(srcs, text_embed, masks, poses, query_embeds)

        # prediction
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
            outputs_coord = tmp.sigmoid()  # cxcywh, range in [0,1]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)  # [4, B, Q, 1]
        outputs_coord = torch.stack(outputs_coords)  # [4, B, Q, 4]

        # rearrange
        outputs_class = rearrange(outputs_class, 'l (b t) q k -> l b t q k', b=b, t=t)  # [4, B, 1, Q, 1]
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l b t q n', b=b, t=t)  # [4, B, 1, Q, 4]

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

