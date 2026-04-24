from torch import nn


class DFormerHead(nn.Module):
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 embedding_dim=256,
                 num_classes=7,
                 **kwargs):
        super(DFormerHead, self).__init__()

    def forward(self, inputs):
        pass