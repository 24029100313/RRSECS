import torch.nn as nn


class P2H(nn.Module):
    '''
    Convert plain features from ViT-based models to multi-scale hierarchical features
    '''
    def __init__(self, in_channels, **kwargs):
        super(P2H, self).__init__()

        out_channels = [in_channels[-1] // 2 ** (i - 1) for i in range(len(in_channels), 0, -1)]

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels[-1], out_channel, 1, 1, 0) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, 4, 0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, 2, 0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, 2, 1)
        ])


    def forward(self, inputs):
        outs = []
        for i, x in enumerate(inputs):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            outs.append(x)

        return outs