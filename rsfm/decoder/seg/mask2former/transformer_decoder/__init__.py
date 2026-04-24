from .mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder


__all__ = ['build_transformer_decoder']


def build_transformer_decoder(num_classes=2, in_channels=256, hidden_dim=256, mask_dim=256,
                              num_queries=100, num_heads=8, dim_feedforward=2048, num_dec_layers=10,
                              mask_classification=True, enforce_input_project=False):

    predictor = MultiScaleMaskedTransformerDecoder(in_channels=in_channels,
                                                   num_classes=num_classes,
                                                   mask_classification=mask_classification,
                                                   hidden_dim=hidden_dim,
                                                   num_queries=num_queries,
                                                   nheads=num_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dec_layers=num_dec_layers - 1,
                                                   pre_norm=False,
                                                   mask_dim=mask_dim,
                                                   enforce_input_project=enforce_input_project)

    return predictor