import torch.nn as nn
from rsfm.decoder.seg.mask2former import Mask2FormerHead
from rsfm.decoder.seg.mask2former.utils import init_feats_dict
from .utils import MultiScaleMaskedReferringDecoder



class ReLAHead(Mask2FormerHead):
    def __init__(self, rla_weight=0.1, embedding_dim=256, num_classes=2, **kwargs):
        super(ReLAHead, self).__init__(**kwargs)

        self.rla_weight = rla_weight

        self.predictor = MultiScaleMaskedReferringDecoder(num_classes=num_classes,
                                                          rla_weight=rla_weight,
                                                          in_channels=embedding_dim,
                                                          hidden_dim=embedding_dim,
                                                          mask_dim=embedding_dim)

        self.conv_seg = nn.Conv2d(num_classes + 1, 2, 1)


    def forward(self, features, lang_feat):
        outputs = init_feats_dict(self.input_shape, features)

        mask_features, _, multi_scale_features = self.pixel_decoder(outputs)
        predictions = self.predictor(multi_scale_features, mask_features, lang_feat)

        out = self.conv_seg(predictions['pred_masks'])

        return out
