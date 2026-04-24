import torch.nn as nn
from rsfm.decoder.seg.mask2former.pixel_decoder import build_pixel_decoder
from rsfm.decoder.seg.mask2former.utils import init_input_dict, init_feats_dict



class RefSegformerHead(nn.Module):
    def __init__(self,
                 in_index=[0, 1, 2, 3],
                 in_channels=[128, 256, 512, 1024],
                 feature_strides=[4, 8, 16, 32],
                 dropout_ratio=0.,
                 num_enc_layers=6,
                 num_heads=8,
                 embedding_dim=256,
                 num_classes=2,
                 **kwargs):
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
        self.mask_conv = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, inputs):
        outputs = init_feats_dict(self.input_shape, inputs)
        mask_features, _, _ = self.pixel_decoder(outputs)
        out = self.mask_conv(mask_features)

        return out



