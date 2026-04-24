from .msdeformattn import MSDeformAttnPixelDecoder


__all__ = ['build_pixel_decoder']


def build_pixel_decoder(common_stride=4, num_heads=8, dropout=0., num_enc_layers=4, conv_dim=256,
                        mask_dim=256, transformer_in_features=["s1", "s2", "s3", "s4"], input_shape=None):

    pixel_decoder = MSDeformAttnPixelDecoder(input_shape=input_shape,
                                             transformer_dropout=dropout,
                                             transformer_nheads=num_heads,
                                             transformer_dim_feedforward=1024,
                                             transformer_enc_layers=num_enc_layers,
                                             conv_dim=conv_dim,
                                             mask_dim=mask_dim,
                                             transformer_in_features=transformer_in_features,
                                             common_stride=common_stride)

    return pixel_decoder
