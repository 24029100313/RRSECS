import torch.nn as nn
from rsfm.decoder.seg.mask2former import Mask2FormerHead
from rsfm.decoder.seg.mask2former.utils import init_feats_dict


__all__ = ['MagNetHead']


class MagNetHead(Mask2FormerHead):
    def __init__(self, **kwargs):
        super(MagNetHead, self).__init__(num_queries=1, num_enc_layers=4, **kwargs)

        self.conv_seg = nn.Conv2d(1, 2, 1)

    def forward(self, features):
        '''
        return: predictions -> dict
                predictions['pred_logits']: B, Q, num_classes + 1
                predictions['pred_masks']: B, Q, H/32, W/32
        '''
        outputs = init_feats_dict(self.input_shape, features)

        mask_features, _, multi_scale_features = self.pixel_decoder(outputs)
        predictions = self.predictor(multi_scale_features, mask_features)

        out = self.conv_seg(predictions['pred_masks'])

        return out


# class MagNetHead(Mask2FormerHead):
#     def __init__(self, **kwargs):
#         super(MagNetHead, self).__init__(num_queries=1, num_enc_layers=4, **kwargs)
#
#     def forward(self, features):
#         '''
#         return: predictions -> dict
#                 predictions['pred_logits']: B, Q, num_classes + 1
#                 predictions['pred_masks']: B, Q, H/32, W/32
#         '''
#         outputs = init_feats_dict(self.input_shape, features)
#
#         mask_features, _, multi_scale_features = self.pixel_decoder(outputs)
#         predictions = self.predictor(multi_scale_features, mask_features)
#
#         return predictions['pred_masks']