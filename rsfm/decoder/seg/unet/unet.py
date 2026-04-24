import torch.nn as nn
import torch.nn.functional as F
from .utils import CenterBlock, DecoderBlock



class UNetHead(nn.Module):
    def __init__(
            self,
            in_channels=[96, 192, 384, 768],
            num_classes=7,
            embedding_dim=64,
            n_blocks=4,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            dropout_ratio=0.1,
            **kwargs
    ):
        super().__init__()

        encoder_channels = in_channels
        decoder_channels = [embedding_dim * 2 ** i for i in range(len(in_channels))][::-1]

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[::-1]  # (2048, 1024, 512, 256, 64); (512, 256, 128 ,64)

        # computing blocks input and output channels
        head_channels = encoder_channels[0]  # 2048; 512
        in_channels = [head_channels] + list(decoder_channels[:-1])  # [2048, 256, 128, 64, 32]; [512, 256, 128, 64]
        skip_channels = list(encoder_channels[1:]) + [0]  # [1024, 512, 256, 64, 0]; [256, 128, 64, 0]
        out_channels = decoder_channels  # (256, 128, 64, 32, 16) # (256, 128, 64, 32)

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm, **kwargs)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type, norm_cfg=norm_cfg)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features):

        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        x = self.dropout(x)
        x = self.conv_seg(x)

        return x