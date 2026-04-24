import torch
import math
import torch.nn as nn
from rsfm.decoder.seg.mask2former import Mask2FormerHead
from rsfm.decoder.seg.mask2former.transformer_decoder import MultiScaleMaskedTransformerDecoder
from rsfm.decoder.seg.mask2former.utils import init_feats_dict
from rsfm.utils import MLP, get_clones, inverse_sigmoid
from .utils import MSDeformAttnTransformerVG, LAGD, MBVIM



class CCFormerHead(Mask2FormerHead):
    def __init__(self,
                 num_classes=1,
                 embedding_dim=256,
                 **kwargs):
        super(CCFormerHead, self).__init__(num_queries=1,
                                           num_enc_layers=4,
                                           embedding_dim=embedding_dim,
                                           num_classes=num_classes,
                                           **kwargs)

        self.gate_decoupler = nn.ModuleList([LAGD(embedding_dim, 768) for _ in range(4)])

        self.predictor_vg = MSDeformAttnTransformerVG(d_model=embedding_dim,
                                                      nhead=8,
                                                      num_decoder_layers=4,
                                                      dim_feedforward=1024,
                                                      dropout=0.1,
                                                      activation="relu",
                                                      return_intermediate_dec=True,
                                                      num_feature_levels=4,
                                                      dec_n_points=4,
                                                      two_stage_num_proposals=1)
        self.class_embed = nn.Linear(embedding_dim, num_classes)
        self.bbox_embed = MLP(embedding_dim, embedding_dim, 4, 3)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        num_pred = self.predictor_vg.decoder.num_layers + 1
        self.class_embed = get_clones(self.class_embed, num_pred)
        self.bbox_embed = get_clones(self.bbox_embed, num_pred)
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        self.predictor_vg.decoder.bbox_embed = self.bbox_embed
        self.predictor_vg.decoder.class_embed = self.class_embed
        for box_embed in self.bbox_embed:
            nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)


    def forward(self, features, l, l_mask):
        outputs = init_feats_dict(self.input_shape, features)

        mask_features, all_features, multi_scale_features = self.pixel_decoder(outputs)
        '''
        mask_features: B, 256, H/4, W/4 (F1)
        all_features: [F4, F3, F2, F1]
        multi_scale_features: [F4, F3, F2]
        '''

        ris_feats, vg_feats = [], []
        for ind, feat in enumerate(all_features):
            ris_feat, vg_feat = self.gate_decoupler[ind](feat, l, l_mask)
            ris_feats.append(ris_feat)
            vg_feats.append(vg_feat)

        # Predict for RIS
        predictions = self.predictor(ris_feats[0:3], ris_feats[-1])
        pred_masks = predictions['pred_masks']

        # Predict for VG
        hs, init_reference, inter_references, _, _ = self.predictor_vg(vg_feats[::-1])
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

        out_vg = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.training:
            out_vg['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        else:
            out_vg = self.out_online_eval(out_vg)

        out = {'pred_masks': pred_masks,
               'pred_bboxs': out_vg}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def out_online_eval(self, outputs):
        assert outputs["pred_boxes"].shape[0] == 1

        pred_logits = outputs["pred_logits"][0]
        pred_bbox = outputs["pred_boxes"][0]
        pred_score = pred_logits.sigmoid()  # [q, k]
        max_score, _ = pred_score.max(-1)  # [q,]
        _, max_ind = max_score.max(-1)  # [1,] # which query
        pred_bbox = pred_bbox[max_ind].unsqueeze(0)  # [xc, yc, w_b, h_b]

        return pred_bbox
