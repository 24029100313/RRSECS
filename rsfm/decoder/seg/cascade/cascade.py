import torch
import torch.nn as nn
from .utils import conv_block, up_conv, Attention_block, SpatialAttention, ChannelAttention



class CASCADEHead(nn.Module):
    '''
    from 'Medical Image Segmentation via Cascaded Attention Decoding'
    '''
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 embedding_dim=256,
                 num_classes=2,
                 dropout_ratio=0.1,
                 **kwargs):
        super(CASCADEHead, self).__init__()

        channels = in_channels[::-1]

        self.Conv_1x1 = nn.Conv2d(channels[0], channels[0], kernel_size=1, stride=1, padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])

        self.Up3 = up_conv(ch_in=channels[0], ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1], F_l=channels[1], F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=2 * channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1], ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2], F_l=channels[2], F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=2 * channels[2], ch_out=channels[2])

        self.Up1 = up_conv(ch_in=channels[2], ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3], F_l=channels[3], F_int=32)
        self.ConvBlock1 = conv_block(ch_in=2 * channels[3], ch_out=channels[3])

        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(2 * channels[1])
        self.CA2 = ChannelAttention(2 * channels[2])
        self.CA1 = ChannelAttention(2 * channels[3])

        self.SA = SpatialAttention()

        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(channels[3], num_classes, kernel_size=1)

    def forward(self, inputs):
        x1, x2, x3, x4 = inputs

        d4 = self.Conv_1x1(x4)

        # CAM4
        d4 = self.CA4(d4) * d4
        d4 = self.SA(d4) * d4
        d4 = self.ConvBlock4(d4)

        # upconv3
        d3 = self.Up3(d4)

        # AG3
        x3 = self.AG3(g=d3, x=x3)

        # Concat 3
        d3 = torch.cat((x3, d3), dim=1)

        # CAM3
        d3 = self.CA3(d3) * d3
        d3 = self.SA(d3) * d3
        d3 = self.ConvBlock3(d3)

        # upconv2
        d2 = self.Up2(d3)

        # AG2
        x2 = self.AG2(g=d2, x=x2)

        # Concat 2
        d2 = torch.cat((x2, d2), dim=1)

        # CAM2
        d2 = self.CA2(d2) * d2
        d2 = self.SA(d2) * d2
        d2 = self.ConvBlock2(d2)

        # upconv1
        d1 = self.Up1(d2)

        # AG1
        x1 = self.AG1(g=d1, x=x1)

        # Concat 1
        d1 = torch.cat((x1, d1), dim=1)

        # CAM1
        d1 = self.CA1(d1) * d1
        d1 = self.SA(d1) * d1
        d1 = self.ConvBlock1(d1)

        x = self.dropout(d1)
        x = self.conv_seg(x)

        return x