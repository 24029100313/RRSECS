import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from rsfm.module.neck.utils import CBR, PBR



class MCM(nn.Module):
    '''
    Multi-scale Convolution Module
    '''
    def __init__(self, in_channels, out_channels):
        super(MCM, self).__init__()

        self.mcm = nn.ModuleList([
            CBR(in_channels, out_channels, kernel_size=1),
            CBR(in_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=True),
            CBR(in_channels, out_channels, kernel_size=5, padding=2, groups=out_channels, bias=True),
            PBR(in_channels, kernel_size=3, padding=1, type='avg'),
            PBR(in_channels, kernel_size=3, padding=1, type='max'),
        ])

        self.fusion = CBR((len(self.mcm) + 1) * in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        outs = [x]
        for mcm in self.mcm:
            out = mcm(x)
            outs.append(out)

        xms = torch.cat((outs), dim=1)
        xms = self.fusion(xms)

        return xms


class CrossAttentionBlock(nn.Module):
    def __init__(self,
                 dim=96,
                 l_dim=768,
                 num_heads=8,
                 dropout=0.,
                 window=16,
                 mlp_ratio=4.):
        super(CrossAttentionBlock, self).__init__()

        # Cross-attention for visual-language feature fusion
        self.norm_ca_vis = nn.LayerNorm(dim)
        self.norm_ca_lang = nn.LayerNorm(l_dim)
        self.cam = CrossAttentionModule(dim, l_dim, num_heads, dropout)

        # Self-attention for multimodal feature enhancement
        self.norm_sa_mm = nn.LayerNorm(dim)
        self.sam = SelfAttentionModule(dim, num_heads, dropout, window)

        # FFN
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x, l, l_mask):
        x_res = x.clone()

        mm = x_res + self.cam(self.norm_ca_vis(x), self.norm_ca_lang(l), l_mask)
        emm = mm + self.sam(self.norm_sa_mm(mm))
        out = emm + self.ffn(self.norm_ffn(emm))

        return out


class CrossAttentionModule(nn.Module):
    def __init__(self, dim, l_dim, num_heads, dropout):
        super(CrossAttentionModule, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_vis = nn.Linear(dim, dim)
        self.k_lang = nn.Linear(l_dim, dim)
        self.v_lang = nn.Linear(l_dim, dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_mm = nn.Linear(dim, dim)
        self.proj_mm_drop = nn.Dropout(dropout)

    def forward(self, x, l, l_mask=None):
        B, N, C = x.shape
        x_res = x.clone()

        q = self.q_vis(x)
        k = self.k_lang(l)
        v = self.v_lang(l)

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
        mm = self.proj_mm(mm)
        mm = self.proj_mm_drop(mm)

        out = x_res * mm

        return out


class SelfAttentionModule(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=8,
                 dropout=0.,
                 window=16,
                 qkv_bias=True,
                 agent_num=49):
        super(SelfAttentionModule, self).__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num
        self.window = window

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)

        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)

        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    def forward(self, x):
        x_res = x.clone()

        b, n, c = x.shape
        h = w = int(n ** 0.5)

        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)

        agent_tokens = self.pool(q.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)

        q = q.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        position_bias1 = F.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
        position_bias1 = position_bias1.reshape(1, self.num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, self.num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2

        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = F.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, self.num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, self.num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2

        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)

        out = x_res * x

        return out


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x