import torch
from torch import nn
from einops import rearrange


class MuModuleList(nn.ModuleList):
    def forward(self, x, mu):
        for layer in self:
            if type(layer) == DynamicLinear:
                x = layer(x, mu)
            else:
                x = layer(x)
        return x


class ParamDecoder(nn.Module):
    def __init__(self, mu_dim, need_in_dim, need_out_dim, k=30):
        super(ParamDecoder, self).__init__()
        self.need_in_dim = need_in_dim
        self.need_out_dim = need_out_dim
        self.k = k
        self.decoder = nn.Linear(mu_dim, need_in_dim * k)
        self.V = nn.parameter.Parameter(torch.zeros(k, need_out_dim))

    def forward(self, t_feat):
        B = t_feat.shape[0]
        U = self.decoder(t_feat).reshape(B, self.need_in_dim, self.k)  # B x need_in_dim x k
        param = torch.einsum('bik,kj->bij', U, self.V).reshape(B, -1)
        return param


class DynamicLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, mu_dim: int, bias=True):
        super(DynamicLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mu_dim = mu_dim
        self.bias = bias
        self.decoder = ParamDecoder(mu_dim, in_dim + 1, out_dim)

    def forward(self, x, mu):
        param = rearrange(self.decoder(mu), 'B (dim_A dim_B) -> B dim_A dim_B', dim_A=self.in_dim + 1,
                          dim_B=self.out_dim)
        weight = param[:, :-1, :]
        bias = param[:, -1, :]
        x = torch.einsum('b...d,bde->b...e', x, weight)
        if self.bias:
            bias = bias.view(((bias.shape[0],) + (1,) * (len(x.size()) - 2) + (bias.shape[-1],)))
            x = x + bias
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels ,text_dim, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = MuModuleList([
            DynamicLinear(gate_channels, gate_channels // reduction_ratio ,text_dim),
            nn.ReLU(),
            DynamicLinear(gate_channels // reduction_ratio, gate_channels ,text_dim)
        ])
        self.pool_types = pool_types
    def forward(self, x ,mu):
        B = x.shape[0]  # batchsize
        D = x.shape[-1]  # dimension
        channel_att_sum = None
        for pool_type in self.pool_types:
            pre_pool = x.view(B, -1, D)
            if pool_type == 'avg':
                avg_pool = torch.mean(pre_pool, dim=1)
                channel_att_raw = self.mlp(avg_pool, mu)
            elif pool_type == 'max':
                max_pool = torch.max(pre_pool, dim=1).values
                channel_att_raw = self.mlp(max_pool, mu)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum)
        scale = scale.view(((scale.shape[0],) + (1,) * (len(x.size()) - 2) + (scale.shape[-1],)))

        return x * scale


class SpatialGate(nn.Module):
    def __init__(self, gate_channels, mu_dim):
        super(SpatialGate, self).__init__()
        self.spatial = DynamicLinear(gate_channels, 1, mu_dim)

    def forward(self, x, mu):
        assert len(x.size()) > 2  # B spatial D

        x_out = self.spatial(x, mu)
        scale = torch.sigmoid(x_out)  # broadcasting
        res = x * scale
        return res


class QueryDynamicAttention(nn.Module):
    def __init__(self, gate_channels, mu_dim, reduction_ratio, pool_types, use_spatial=True, use_channel=True):
        super(QueryDynamicAttention, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, mu_dim, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate(gate_channels, mu_dim)
        self.use_spatial = use_spatial
        self.use_channel = use_channel

    def forward(self, x, mu):
        if self.use_channel:
            x = self.ChannelGate(x, mu)
        if len(x.size()) <= 2:
            return x
        if self.use_spatial:
            x = self.SpatialGate(x, mu)
        return x