import torch
import torch.nn as nn


class MDCBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        custom_params={
            "kernel": (3, 3, 3),
            "padding": (3, 5, 7),
            "dilation": (3, 5, 7),
        },
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.custom_params = custom_params
        self.kernel = self.custom_params["kernel"]
        self.paddings = self.custom_params["padding"]
        self.dilations = self.custom_params["dilation"]
        SPLIT_NUM = 3

        self.layers = nn.ModuleList()

        self.pre_conv_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            bias=False,
        )
        quotient = self.in_channels // SPLIT_NUM
        reminder = self.in_channels % SPLIT_NUM
        sprit_channels = [quotient] * SPLIT_NUM
        if reminder == 1:
            sprit_channels[0] += 1
            sprit_channels[1] += 1
            sprit_channels[2] -= 1
        elif reminder == 2:
            sprit_channels[0] += 1
            sprit_channels[1] += 1
        for kernel, padding, dilation, channels in zip(
            *custom_params.values(), sprit_channels
        ):
            self.layers.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                )
            )

        self.fusion_layer = nn.Conv2d(
            in_channels=self.in_channels,  # equals to out_channels*2
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x_shape = x.shape
        x = self.pre_conv_layer(x)
        x1, x2, x3 = torch.chunk(x, 3, dim=1)

        assert (
            x1.shape[1] + x2.shape[1] + x3.shape[1] == x_shape[1]
        ), f"{x1.shape[1]} + {x2.shape[1]} + {x3.shape[1]} != {x_shape[1]}"

        x1 = self.layers[0](x1)
        x2 = self.layers[1](x2)
        x3 = self.layers[2](x3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fusion_layer(x)

        return self.act(self.norm(x))


class up_pooling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()

        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_up(x)