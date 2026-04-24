from torch import nn
import torch
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(CrossModalAttention, self).__init__()

        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads

        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        self.project_k = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.project_q = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.project_v = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )


    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, C_v)
        # l input shape: (B, C_l, T)
        # l_mask shape: (B, T, 1)

        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, T, 1) -> (B, 1, T)

        query = self.project_q(x)  # (B, key_channels, H*W)
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.project_k(l)  # (B, key_channels, T)
        value = self.project_v(l)  # (B, value_channels, T)

        key = key * l_mask  # (B, key_channels, T)
        value = value * l_mask  # (B, value_channels, T)
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels // self.num_heads).permute(0, 2, 1, 3)
        # (B, num_heads, H*W, key_channels//num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels // self.num_heads, n_l)
        # (B, num_heads, key_channels//num_heads, T)
        value = value.reshape(B, self.num_heads, self.value_channels // self.num_heads, n_l)
        # # (B, num_heads, value_channels//num_heads, T)
        l_mask = l_mask.unsqueeze(1)  # (B, 1, 1, T)

        # attention score
        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, T)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4 * l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, T)

        # visual-linguistic correlation
        sim_temp = torch.sum(sim_map, dim=3)

        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, out_channels, HW)
        out = out.permute(0, 2, 1)  # (B, HW, out_channels)

        return out, sim_temp


class GatedCrossModalAttention(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(GatedCrossModalAttention, self).__init__()

        self.project_v = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),
                                       nn.GELU(),
                                       nn.Dropout(dropout)
                                       )

        self.vis_lang_att = CrossModalAttention(v_in_channels,  # v_in
                                                l_in_channels,  # l_in
                                                key_channels,  # key
                                                value_channels,  # value
                                                out_channels=value_channels,  # out
                                                num_heads=num_heads
                                                )

        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )


    def forward(self, x, l, l_mask):
        cross, sim_temp = self.vis_lang_att(x, l, l_mask)  # (B, H*W, C_v)

        gated_cross = self.res_gate(cross) * cross

        return gated_cross, sim_temp


#####################################################################################################################


class SpatialCrossScaleAtten(nn.Module):
    def __init__(self, dim):
        super(SpatialCrossScaleAtten, self).__init__()
        self.qkv_linear = nn.Linear(dim, dim * 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.num_head = 8
        self.scale = (dim // self.num_head) ** 0.5

    def forward(self, x):
        B, num_blocks, _, C = x.shape  # (B, K, N, C)

        # (3, B, K, head, N, C)
        qkv = self.qkv_linear(x).reshape(B, num_blocks, -1, 3, self.num_head, C // self.num_head).permute(3, 0, 1, 4, 2,
                                                                                                          5).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        atten = q @ k.transpose(-1, -2).contiguous()
        atten = self.softmax(atten)

        atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, -1, C)
        atten_value = self.proj(atten_value)  # (B, K, K, N, C) / (B, K, N, C)

        return atten_value


class CrossScaleEnhance(nn.Module):
    def __init__(self, dim):
        super(CrossScaleEnhance, self).__init__()
        self.Attention = SpatialCrossScaleAtten(dim)
        layer_scale_init_value = 1
        self.layer_scale = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        h = x  # (B, N, H)
        x = self.Attention(x)
        x = h + self.layer_scale * x

        return x


class URCE(nn.Module):
    def __init__(self, dim=256, channels=[64, 128, 320, 512], num=1):  # dim = 256
        super(URCE, self).__init__()
        self.ini_win_size = 2
        self.channels = channels
        self.dim = dim
        self.fc_module = nn.ModuleList()
        self.fc_rever_module = nn.ModuleList()
        self.num = num
        self.num_stages = 4
        self.topK = 32

        for i in range(self.num_stages):
            self.fc_module.append(nn.Linear(self.channels[i], self.dim))

        for i in range(self.num_stages):
            self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))

        self.group_attention = []
        for i in range(self.num):
            self.group_attention.append(CrossScaleEnhance(dim))
        self.group_attention = nn.Sequential(*self.group_attention)

        self.split_list = [8 * 8, 4 * 4, 2 * 2, 1 * 1]

    def forward(self, x, sim_temp):

        # Uncertain Region Extraction
        s1, s2, s3, s4 = sim_temp
        x1, x2, x3, x4 = x
        B, C, H, W = x1.shape

        h = H // (self.ini_win_size ** (self.num_stages - 1))
        w = W // (self.ini_win_size ** (self.num_stages - 1))
        map_U = torch.zeros((B, 1, h, w), device=x1.device)

        s1 = s1.reshape(B, 1, H // (self.ini_win_size ** 0), W // (self.ini_win_size ** 0))
        s2 = s2.reshape(B, 1, H // (self.ini_win_size ** 1), W // (self.ini_win_size ** 1))
        s3 = s3.reshape(B, 1, H // (self.ini_win_size ** 2), W // (self.ini_win_size ** 2))
        s4 = s4.reshape(B, 1, H // (self.ini_win_size ** 3), W // (self.ini_win_size ** 3))
        s1 = F.interpolate(input=s1, size=(h, w), mode='bilinear', align_corners=True)
        s2 = F.interpolate(input=s2, size=(h, w), mode='bilinear', align_corners=True)
        s3 = F.interpolate(input=s3, size=(h, w), mode='bilinear', align_corners=True)

        map_U = (s1 - s2).abs() + (s2 - s3).abs() + (s3 - s4).abs()

        topk_score, topk_idx = torch.topk(map_U.reshape(B, 1, h * w), dim=-1, k=self.topK, largest=True)

        # Cross-Scale Fusing Attention
        # channel unify
        x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H_i, W_i, dim)]

        # patch merging
        N = 0
        for j, item in enumerate(x):
            B, H, W, C = item.shape
            win_size = self.ini_win_size ** (self.num_stages - j - 1)
            item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(0, 1, 3, 2, 4,
                                                                                                5).contiguous()
            item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
            N = N + win_size * win_size
            x[j] = item

        x = tuple(x)
        x = torch.cat(x, dim=-2)  # (B, h, w, N, dim)
        y = torch.zeros((B, self.topK, N, self.dim), device=x.device)  # (B, K, N, dim)

        # y: cross-scale features of uncertain regions
        for j in range(self.topK):
            for i, item in enumerate(topk_idx):
                it = item[0][j]
                xt, yt = it // h, it % w
                y[i][j] = x[i][xt][yt]

        # multi-head self attention with spatial correspondence
        for i in range(self.num):
            y = self.group_attention[i](y)  # (B, K, N, dim)

        x = 2 * x
        for i, item in enumerate(topk_idx):
            for j in range(self.topK):
                it = item[0][j]
                xt, yt = it // h, it % w
                x[i][xt][yt] = y[i][j]

        # patch reversion
        x = torch.split(x, self.split_list, dim=-2)
        x = list(x)

        for j, item in enumerate(x):
            B, num_blocks, _, N, C = item.shape
            win_size = self.ini_win_size ** (self.num_stages - j - 1)
            item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(0, 1, 3, 2, 4,
                                                                                          5).contiguous().reshape(B,
                                                                                                                  num_blocks * win_size,
                                                                                                                  num_blocks * win_size,
                                                                                                                  C)
            item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
            x[j] = item

        return x