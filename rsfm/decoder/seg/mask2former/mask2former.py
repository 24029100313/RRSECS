from torch import nn
from .pixel_decoder import build_pixel_decoder
from .transformer_decoder import build_transformer_decoder
from .utils import init_input_dict, init_feats_dict


# TODO not working now !!!

class Mask2FormerHead(nn.Module):
    def __init__(self,
                 in_index=[0, 1, 2, 3],
                 in_channels=[128, 256, 512, 1024],
                 feature_strides=[4, 8, 16, 32],
                 dropout_ratio=0.,
                 num_enc_layers=6,
                 num_heads=8,
                 embedding_dim=256,
                 num_queries=100,
                 num_classes=2,
                 dim_feedforward=2048,
                 num_dec_layers=10,
                 mask_classification=True,
                 **kwargs
):
        super().__init__()

        input_shape = init_input_dict(in_index, in_channels, feature_strides)
        self.input_shape = input_shape

        self.pixel_decoder = build_pixel_decoder(num_heads=num_heads,
                                                 num_enc_layers=num_enc_layers,
                                                 conv_dim=embedding_dim,
                                                 mask_dim=embedding_dim,
                                                 dropout=dropout_ratio,
                                                 input_shape=input_shape,
                                                 transformer_in_features=[k for k in input_shape.keys()])

        self.predictor = build_transformer_decoder(num_classes=num_classes,
                                                   in_channels=embedding_dim,
                                                   hidden_dim=embedding_dim,
                                                   mask_dim=embedding_dim,
                                                   num_queries=num_queries,
                                                   num_heads=num_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   num_dec_layers=num_dec_layers,
                                                   mask_classification=mask_classification)

    def forward(self, features):
        '''
        return: predictions -> dict
                predictions['pred_logits']: B, Q, num_classes + 1
                predictions['pred_masks']: B, Q, H/32, W/32
        '''
        outputs = init_feats_dict(self.input_shape, features)

        mask_features, _, multi_scale_features = self.pixel_decoder(outputs)
        predictions = self.predictor(multi_scale_features, mask_features)

        return predictions