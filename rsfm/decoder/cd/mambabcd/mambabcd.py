import torch.nn as nn
from .utils import ChangeDecoder


class MambaBCDHead(nn.Module):
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 embedding_dim=128,
                 num_classes=2, **kwargs):
        super(MambaBCDHead, self).__init__()

        self.decoder = ChangeDecoder(encoder_dims=in_channels, channel_first=True)
        self.main_clf = nn.Conv2d(in_channels=embedding_dim, out_channels=num_classes, kernel_size=1)

    def forward(self, features):
        pre_features, post_features = [], []
        for i in range(len(features)):
            pre_feat, post_feat = features[i].chunk(2)
            pre_features.append(pre_feat)
            post_features.append(post_feat)

        # Decoder processing - passing encoder outputs to the decoder
        output = self.decoder(pre_features, post_features)
        output = self.main_clf(output)
        return output