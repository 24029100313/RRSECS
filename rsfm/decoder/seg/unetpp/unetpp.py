import torch
from torch import nn
import torch.nn.functional as F
from rsfm.decoder.seg.unet.utils import DecoderBlock, CenterBlock



class UNetPPHead(nn.Module):
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
        self.in_channels = [head_channels] + list(decoder_channels[:-1])  # [2048, 256, 128, 64, 32]; [512, 256, 128, 64]
        self.skip_channels = list(encoder_channels[1:]) + [0]  # [1024, 512, 256, 64, 0]; [256, 128, 64, 0]
        self.out_channels = decoder_channels  # (256, 128, 64, 32, 16) # (256, 128, 64, 32)

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm, **kwargs)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type, norm_cfg=norm_cfg)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (
                            layer_idx + 1 - depth_idx
                    )
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(
                    in_ch, skip_ch, out_ch, **kwargs
                )
        blocks[f"x_{0}_{len(self.in_channels) - 1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features):
        features = features[::-1]  # reverse channels to start from head of encoder

        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](
                        features[depth_idx], features[depth_idx + 1]
                    )
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [
                        dense_x[f"x_{idx}_{dense_l_i}"]
                        for idx in range(depth_idx + 1, dense_l_i + 1)
                    ]
                    cat_features = torch.cat(
                        cat_features + [features[dense_l_i + 1]], dim=1
                    )
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[
                        f"x_{depth_idx}_{dense_l_i}"
                    ](dense_x[f"x_{depth_idx}_{dense_l_i - 1}"], cat_features)
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](
            dense_x[f"x_{0}_{self.depth - 1}"]
        )
        x = dense_x[f"x_{0}_{self.depth}"]

        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        x = self.dropout(x)
        x = self.conv_seg(x)

        return x