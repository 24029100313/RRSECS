import torch.nn as nn
from rsfm.decoder.seg.mask2former import Mask2FormerHead
from rsfm.decoder.seg.mask2former.utils import init_feats_dict
from .utils import LAGD



class CCFormerRISHead(Mask2FormerHead):
    def __init__(self,
                 num_classes=1,
                 embedding_dim=256,
                 **kwargs):
        super(CCFormerRISHead, self).__init__(num_queries=1,
                                              num_enc_layers=4,
                                              embedding_dim=embedding_dim,
                                              num_classes=num_classes,
                                              **kwargs)

        self.gate_decoupler = nn.ModuleList([LAGD(embedding_dim, 768) for _ in range(4)])

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

        return pred_masks