import torch
import torch.nn as nn
from torch.nn import BatchNorm2d
from torch.nn import Module, Conv2d, Parameter
from rsfm.module.neck.utils import CBR


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class Attention(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(Attention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (self.gamma * weight_value).contiguous()
    

class Output(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(Output, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes * up_factor * up_factor
        self.conv = CBR(in_chan, mid_chan, 3, 1, 1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.PixelShuffle(up_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x


class AttentionEnhancementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionEnhancementModule, self).__init__()
        self.conv = CBR(in_chan, out_chan, 3, 1, 1)
        self.conv_atten = Attention(out_chan)
        self.bn_atten = BatchNorm2d(out_chan)

    def forward(self, x):
        feat = self.conv(x)
        att = self.conv_atten(feat)
        return self.bn_atten(att)


class ContextPath(nn.Module):
    def __init__(self, in_chan1, in_chan2):
        super(ContextPath, self).__init__()
        self.arm16 = AttentionEnhancementModule(in_chan1, 128)   #256-384
        self.arm32 = AttentionEnhancementModule(in_chan2, 128)  #512-768
        self.conv_head32 = CBR(128, 128, 3, 1, 1)
        self.conv_head16 = CBR(128, 128, 3, 1, 1)
        self.conv_avg = CBR(in_chan2, 128, 1) #512-768
        self.up32 = nn.Upsample(scale_factor=2.)
        self.up16 = nn.Upsample(scale_factor=2.)


    def forward(self, inputs):

        feat16 = inputs[2]
        feat32 = inputs[3]

        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = self.up32(feat32_sum)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum)
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up  # x8, x16


class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = CBR(3, 64, 7, 2, 3)
        self.conv2 = CBR(64, 64, 3, 2, 1)
        self.conv3 = CBR(64, 64, 3, 2, 1)
        self.conv_out = CBR(64, 128, 1)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat


class FeatureAggregationModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureAggregationModule, self).__init__()
        self.convblk = CBR(in_chan, out_chan, 1)
        self.conv_atten = Attention(out_chan)

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = self.conv_atten(feat)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

