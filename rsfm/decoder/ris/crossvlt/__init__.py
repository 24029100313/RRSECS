from rsfm.decoder.ris.lavt import LAVTHead


__all__ = ['CrossVLTHead']


class CrossVLTHead(LAVTHead):
    def __init__(self,
                 in_channels=[128, 256, 512, 1024],
                 embedding_dim=512,
                 num_classes=2,
                 **kwargs):
        super(CrossVLTHead, self).__init__(in_channels=in_channels,
                                           embedding_dim=embedding_dim,
                                           num_classes=num_classes,
                                           **kwargs)