import torch
from torch import nn


class CBAM(nn.Module):
    '''
    Channel and spatial attention from 'Convolutional Block Attention Module, ECCV 2018'
    '''
    def __init__(self, in_channels, reduction=16, spatial_kernel=7, channel_first=True):
        super(CBAM, self).__init__()
        self.channel_first = channel_first

        self.channel_att = ChannelGate(in_channels, reduction)
        self.spatial_att = SpatialGate(spatial_kernel)

    def forward(self, x):
        '''
        x: B, C, H, W
        '''
        if self.channel_first:
            x = self.spatial_att(self.channel_att(x))
        else:
            x = self.channel_att(self.spatial_att(x))

        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, in_channels, reduction):
        super(ChannelGate, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )

        self.act = nn.Sigmoid()

    def forward(self, x):
        att = self.act(self.mlp(self.max_pool(x)) + self.mlp(self.avg_pool(x))).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * att


class SpatialGate(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialGate, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x_compress =  torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x = x * self.act(self.bn(self.conv(x_compress)))

        return x