from ...seg import Mask2FormerHead


__all__ = ['MagNetHead']

# TODO not working now !!!

class MagNetHead(Mask2FormerHead):
    def __init__(self, **kwargs):
        super(MagNetHead, self).__init__(num_queries=1, num_enc_layers=4, **kwargs)
