import math
import torch
from timm.layers import trunc_normal_
from torch import nn
from .utils import CBR, CGR
from rsfm.module.ris_fusion.lavt_fusion.utils import PWAM
from rsfm.utils import NestedTensor




class CLM(nn.Module):
    '''Channel-wise Language-enhanced Module'''
    def __init__(self, dim, l_dim):
        super(CLM, self).__init__()
        self.vis_proj = CBR(dim, dim)

        self.lang_proj = nn.Sequential(nn.Linear(l_dim, dim),
                                       nn.LayerNorm(dim),
                                       nn.GELU())

        self.lang_channel_weight = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                                 nn.Conv1d(dim, dim //4, kernel_size=1, stride=1),
                                                 nn.ReLU(inplace=True),
                                                 nn.Conv1d(dim // 4, dim, kernel_size=1, stride=1),
                                                 nn.Tanh())

        self.mix_proj = CBR(dim, dim)

    def forward(self, x, l):
        x_res = x.clone()

        x = self.vis_proj(x).permute(0, 2, 3, 1) # b, h, w, dim

        l = self.lang_proj(l.permute(0, 2, 1)) # b, n, dim

        lcw = self.lang_channel_weight(l.permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(1) # b, 1, 1, dim

        mix = x * lcw # b, h, w, dim
        mix = self.mix_proj(mix.permute(0, 3, 1, 2))

        out = x_res + mix

        return out



# class VLF(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=1,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vlf_layers = nn.ModuleList([
#             PWAM(dim, dim, l_dim, dim, dim, num_heads, dropout) for dim in in_channels
#         ])
#
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             B, C, H, W = x.shape
#             x = x.flatten(2).transpose(1, 2)
#             x = self.vlf_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
# V1 Results
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 75.27 | 63.79 | 41.32 | 17.14 |  4.67 | 77.92 | 60.54 |
# +-------+-------+-------+-------+-------+-------+-------+


# class VLF(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=1,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionModule(dim, l_dim, num_heads, dropout) for dim in in_channels
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
# V2 Results
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 78.02 | 67.47 | 48.24 | 21.98 |  6.81 | 80.03 | 63.15 |
# +-------+-------+-------+-------+-------+-------+-------+



# class VLF(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=4,
#                  dropout=0.):
#         super(VLF, self).__init__()
# 
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionModule(dim, l_dim, num_heads, dropout) for dim in in_channels
#         ])
# 
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
# 
#         return outs
# V3 Results
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 77.91 | 67.97 | 48.46 | 21.59 |  7.03 | 80.01 | 63.14 |
# +-------+-------+-------+-------+-------+-------+-------+


# class VLF(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionModule(dim, l_dim, num_heads, dropout) for dim in in_channels
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         x = x.flatten(2).transpose(1, 2)
#         x_res = x.clone()
#         B, N, C = x.shape
#
#         q = self.q_vis(x)
#         k = self.k_lang(l.permute(0, 2, 1))
#         v = self.v_lang(l.permute(0, 2, 1))
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         out = x_res * mm
#
#         return out
# V4 Results, V1-V4 same CAM
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 78.74 |  68.9 |  49.4 | 21.92 |  6.92 | 80.15 | 63.41 |
# +-------+-------+-------+-------+-------+-------+-------+



# class VLF(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionModule(dim, l_dim, num_heads, dropout) for dim in in_channels
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         x = x.flatten(2).transpose(1, 2)
#         x_res = x.clone()
#         B, N, C = x.shape
#
#         q = self.q_vis(x)
#         k = self.k_lang(l.permute(0, 2, 1))
#         v = self.v_lang(l.permute(0, 2, 1))
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         out = x_res + mm
#
#         return out
# V5 Results
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 75.44 | 64.56 | 46.32 | 19.95 |  7.25 | 79.86 | 61.44 |
# +-------+-------+-------+-------+-------+-------+-------+



# class VLF(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionModule(out_channels, l_dim, num_heads, dropout) for _ in in_channels
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
# V6 Results
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 79.18 | 69.62 | 51.65 | 24.01 |  6.92 | 80.33 | 64.04 |
# +-------+-------+-------+-------+-------+-------+-------+



# class VLF(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CGR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionModule(out_channels, l_dim, num_heads, dropout) for _ in in_channels
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs


# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         x = x.flatten(2).transpose(1, 2)
#         x_res = x.clone()
#         B, N, C = x.shape
#
#         q = self.q_vis(x)
#         k = self.k_lang(l.permute(0, 2, 1))
#         v = self.v_lang(l.permute(0, 2, 1))
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         out = x_res * mm
#
#         return out


# V7 Results
# +-------+-------+-------+-------+-------+------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 | oIoU |  mIoU |
# +-------+-------+-------+-------+-------+------+-------+
# |  79.4 | 70.71 | 51.04 | 22.47 |  6.7  | 79.8 | 63.88 |
# +-------+-------+-------+-------+-------+------+-------+



# class VLF(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.,
#                  sr_ratios=[8, 4, 2, 1]):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionModule(out_channels, l_dim, num_heads, dropout) for _ in in_channels
#         ])
#
#         self.self_att_layers = nn.ModuleList([
#             SelfAttentionModule(out_channels, num_heads, dropout=dropout, sr_ratio=sr_ratio) for sr_ratio in sr_ratios
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             x = self.self_att_layers[idx](x)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         x = x.flatten(2).transpose(1, 2)
#         x_res = x.clone()
#         B, N, C = x.shape
#
#         q = self.q_vis(x)
#         k = self.k_lang(l.permute(0, 2, 1))
#         v = self.v_lang(l.permute(0, 2, 1))
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         out = x_res * mm
#
#         return out
#
#
# class SelfAttentionModule(nn.Module):
#     def __init__(self, dim, num_heads, qkv_bias=True, dropout=0., sr_ratio=8):
#         super(SelfAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)
#
#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm = nn.LayerNorm(dim)
#
#     def forward(self, x):
#         x_res = x.clone()
#
#         B, N, C = x.shape
#         H = W = int(math.sqrt(N))
#
#         q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         if self.sr_ratio > 1:
#             x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#             x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
#             x_ = self.norm(x_)
#             kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         else:
#             kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         out = x_res * x
#
#         return out
# V8 Results
# +-------+-------+-------+-------+-------+-------+------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU | mIoU |
# +-------+-------+-------+-------+-------+-------+------+
# | 72.36 | 59.73 | 36.48 | 14.78 |  4.67 | 77.53 | 58.7 |
# +-------+-------+-------+-------+-------+-------+------+




# class VLF(nn.Module):
#     def __init__(self,
#                  img_size=512,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionModule(out_channels, l_dim, num_heads, dropout) for _ in in_channels
#         ])
#
#         self.self_att_layers = nn.ModuleList([
#             SelfAttentionModule(out_channels, num_heads, dropout=dropout, window=img_size // (2 ** (i + 2)))
#             for i in range(len(in_channels))
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             x = self.self_att_layers[idx](x)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         x = x.flatten(2).transpose(1, 2)
#         x_res = x.clone()
#         B, N, C = x.shape
#
#         q = self.q_vis(x)
#         k = self.k_lang(l.permute(0, 2, 1))
#         v = self.v_lang(l.permute(0, 2, 1))
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         out = x_res * mm
#
#         return out
#
#
# class SelfAttentionModule(nn.Module):
#     def __init__(self, dim, num_heads, qkv_bias=True, dropout=0., agent_num=49, window=16):
#         super(SelfAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.agent_num = agent_num
#         self.window = window
#
#         self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
#
#         self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
#         self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
#         self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
#         self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
#         trunc_normal_(self.an_bias, std=.02)
#         trunc_normal_(self.na_bias, std=.02)
#         trunc_normal_(self.ah_bias, std=.02)
#         trunc_normal_(self.aw_bias, std=.02)
#         trunc_normal_(self.ha_bias, std=.02)
#         trunc_normal_(self.wa_bias, std=.02)
#
#         pool_size = int(agent_num ** 0.5)
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
#
#     def forward(self, x):
#         x_res = x.clone()
#
#         B, N, C = x.shape
#         H = W = int(math.sqrt(N))
#
#         qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         q, k, v = qkv.unbind(0)
#
#         agent_tokens = self.pool(q.reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         agent_tokens = agent_tokens.reshape(B, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
#         position_bias1 = position_bias1.reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias = position_bias1 + position_bias2
#
#         agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
#         agent_attn = self.attn_drop(agent_attn)
#         agent_v = agent_attn @ v
#
#         agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
#         agent_bias1 = agent_bias1.reshape(1, self.num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(B, 1, 1, 1)
#         agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, self.num_heads, -1, self.agent_num).repeat(B, 1, 1, 1)
#         agent_bias = agent_bias1 + agent_bias2
#
#         q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
#         q_attn = self.attn_drop(q_attn)
#         x = q_attn @ agent_v
#
#         x = x.transpose(1, 2).reshape(B, N, C)
#         v = v.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
#         x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(B, N, C)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         out = x_res * x
#
#         return out
# V9 Results
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 74.78 | 62.86 |  42.2 | 16.37 |  4.84 | 78.09 | 60.57 |
# +-------+-------+-------+-------+-------+-------+-------+



# class VLF(nn.Module):
#     def __init__(self,
#                  img_size=512,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.self_att_layers = nn.ModuleList([
#             SelfAttentionModule(out_channels, num_heads, dropout=dropout, window=img_size // (2 ** (i + 2)))
#             for i in range(len(in_channels))
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionModule(out_channels, l_dim, num_heads, dropout) for _ in in_channels
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.self_att_layers[idx](x.flatten(2).transpose(1, 2))
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
# class SelfAttentionModule(nn.Module):
#     def __init__(self, dim, num_heads, qkv_bias=True, dropout=0., agent_num=49, window=16):
#         super(SelfAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.agent_num = agent_num
#         self.window = window
#
#         self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
#
#         self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
#         self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
#         self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
#         self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
#         trunc_normal_(self.an_bias, std=.02)
#         trunc_normal_(self.na_bias, std=.02)
#         trunc_normal_(self.ah_bias, std=.02)
#         trunc_normal_(self.aw_bias, std=.02)
#         trunc_normal_(self.ha_bias, std=.02)
#         trunc_normal_(self.wa_bias, std=.02)
#
#         pool_size = int(agent_num ** 0.5)
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
#
#     def forward(self, x):
#         x_res = x.clone()
#
#         B, N, C = x.shape
#         H = W = int(math.sqrt(N))
#
#         qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         q, k, v = qkv.unbind(0)
#
#         agent_tokens = self.pool(q.reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         agent_tokens = agent_tokens.reshape(B, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
#         position_bias1 = position_bias1.reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias = position_bias1 + position_bias2
#
#         agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
#         agent_attn = self.attn_drop(agent_attn)
#         agent_v = agent_attn @ v
#
#         agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
#         agent_bias1 = agent_bias1.reshape(1, self.num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(B, 1, 1, 1)
#         agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, self.num_heads, -1, self.agent_num).repeat(B, 1, 1, 1)
#         agent_bias = agent_bias1 + agent_bias2
#
#         q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
#         q_attn = self.attn_drop(q_attn)
#         x = q_attn @ agent_v
#
#         x = x.transpose(1, 2).reshape(B, N, C)
#         v = v.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
#         x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(B, N, C)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         out = x_res * x
#
#         return out
#
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         x_res = x.clone()
#         B, N, C = x.shape
#
#         q = self.q_vis(x)
#         k = self.k_lang(l.permute(0, 2, 1))
#         v = self.v_lang(l.permute(0, 2, 1))
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         out = x_res * mm
#
#         return out
# V10 Results
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 75.66 | 64.07 | 43.96 | 17.91 |  5.77 | 78.63 | 60.98 |
# +-------+-------+-------+-------+-------+-------+-------+



# class VLF(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionModule(out_channels, l_dim, num_heads, dropout) for _ in in_channels
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
# class PositionEmbeddingSine1D(nn.Module):
#     def __init__(self, num_pos_feats=256, temperature=10000, normalize=False, scale=None):
#         super().__init__()
#         self.num_pos_feats = num_pos_feats
#         self.temperature = temperature
#         self.normalize = normalize
#         if scale is not None and normalize is False:
#             raise ValueError("normalize should be True if scale is passed")
#         if scale is None:
#             scale = 2 * math.pi
#         self.scale = scale
#
#     def forward(self, tensor_list: NestedTensor):
#         x = tensor_list.tensors # [B, N, C]
#         mask = tensor_list.mask # [B, N]
#         assert mask is not None
#         not_mask = ~mask
#         x_embed = not_mask.cumsum(1, dtype=torch.float32)
#         if self.normalize:
#             eps = 1e-6
#             x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale
#
#         dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
#         dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
#
#         pos_x = x_embed[:, :, None] / dim_t
#         pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
#
#         return pos_x # [B, N, C]
#
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.text_pos = PositionEmbeddingSine1D(dim, normalize=True)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         x = x.flatten(2).transpose(1, 2)
#         x_res = x.clone()
#         B, N, C = x.shape
#
#         q = self.q_vis(x)
#         k = self.k_lang(l.permute(0, 2, 1))
#         v = self.v_lang(l.permute(0, 2, 1))
#
#         k = k + self.text_pos(NestedTensor(k, l_mask.squeeze(-1)))
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         out = x_res * mm
#
#         return out
# V100 Results
# +-------+-------+-------+-------+-------+------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 | oIoU |  mIoU |
# +-------+-------+-------+-------+-------+------+-------+
# | 79.67 | 70.22 | 52.97 | 23.63 |  6.92 | 80.2 | 64.33 |
# +-------+-------+-------+-------+-------+------+-------+



# class VLF(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionBlock(out_channels, l_dim, num_heads, dropout) for _ in in_channels
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
# # V12 Results
# # +-------+-------+-------+-------+-------+-------+------+
# # | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU | mIoU |
# # +-------+-------+-------+-------+-------+-------+------+
# # | 79.34 | 70.33 | 54.07 | 25.82 |  7.86 | 80.51 | 64.6 |
# # +-------+-------+-------+-------+-------+-------+------+
#
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         x_res = x.clone()
#         B, N, C = x.shape
#
#         q = self.q_vis(x)
#         k = self.k_lang(l)
#         v = self.v_lang(l)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         out = x_res * mm
#
#         return out
#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
#
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout, mlp_ratio=4.):
#         super(CrossAttentionBlock, self).__init__()
#         self.norm1_vis = nn.LayerNorm(dim)
#         self.norm1_lang = nn.LayerNorm(l_dim)
#         self.cross_att = CrossAttentionModule(dim, l_dim, num_heads, dropout)
#
#         self.norm2_mm = nn.LayerNorm(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
#
#     def forward(self, x, l, l_mask):
#         x = x.flatten(2).transpose(1, 2)
#         x_res = x.clone()
#
#         mm = x_res + self.cross_att(self.norm1_vis(x), self.norm1_lang(l.permute(0, 2, 1)), l_mask)
#         out = mm + self.mlp(self.norm2_mm(mm))
#
#         return out


# class VLF(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionBlock(out_channels, l_dim, num_heads, dropout) for _ in in_channels
#         ])
#
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#
#
#     def forward(self, inputs, l, l_mask):
#         laterals = []
#         for i in range(len(inputs)):
#             x = self.vis_proj_layers[::-1][i](inputs[::-1][i])
#             if i == 0:
#                 lateral = x
#             else:
#                 lateral = x + self.upsample(laterals[-1])
#             laterals.append(lateral)
#
#         outs = []
#         for idx, x in enumerate(laterals[::-1]):
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
# V13 Results
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 78.68 | 69.18 | 49.56 | 22.58 |  6.92 | 80.21 | 63.87 |
# +-------+-------+-------+-------+-------+-------+-------+



# class VLF(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionBlock(out_channels, l_dim, num_heads, dropout) for _ in in_channels
#         ])
#
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#
#         self.mm_fpn_proj_layers = nn.ModuleList([
#             CBR(out_channels, out_channels) for _ in in_channels
#         ])
#
#
#     def forward(self, inputs, l, l_mask):
#         laterals = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             lateral = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             laterals.append(lateral)
#
#         outs = []
#         for i in range(len(laterals)):
#             mm = self.mm_fpn_proj_layers[::-1][i](laterals[::-1][i])
#             if i == 0:
#                 out = mm
#             else:
#                 out = mm + self.upsample(outs[-1])
#             outs.append(out)
#
#         return outs[::-1]
# V14 Results
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 79.73 | 69.45 | 50.88 | 22.97 |  7.47 | 80.31 | 64.43 |
# +-------+-------+-------+-------+-------+-------+-------+



# class VLF(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionBlock(out_channels, l_dim, num_heads, dropout) for _ in in_channels
#         ])
#
#         self.mm_down_layers = nn.ModuleList([
#             CBR(out_channels, out_channels, 3, 2, 1),
#             CBR(out_channels, out_channels, 3, 2, 1),
#             CBR(out_channels, out_channels, 3, 2, 1)
#         ])
#
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         for i in range(0, 3):
#             outs[i + 1] = outs[i + 1] + self.mm_down_layers[i](outs[i])
#
#         return outs
# V15 Results
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 79.07 | 69.78 | 51.37 |  25.0 |  7.31 | 80.05 | 64.02 |
# +-------+-------+-------+-------+-------+-------+-------+



# class VLF(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionBlock(out_channels, l_dim, num_heads, dropout) for _ in in_channels
#         ])
#
#         self.mm_down_layers = nn.ModuleList([
#             CBR(out_channels, out_channels, 3, 2, 1),
#             CBR(out_channels, out_channels, 3, 2, 1),
#             CBR(out_channels, out_channels, 3, 2, 1)
#         ])
#
#         self.mm_proj_layers = nn.ModuleList([
#             CBR(out_channels, out_channels, 3, 1, 1) for _ in in_channels
#         ])
#
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         for i in range(0, 3):
#             outs[i + 1] = outs[i + 1] + self.mm_down_layers[i](outs[i])
#
#         outs = [self.mm_proj_layers[i](out) for i, out in enumerate(outs)]
#
#         return outs
# V16 Results
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 75.16 | 64.56 | 44.95 | 19.84 |  6.1  | 78.65 | 61.21 |
# +-------+-------+-------+-------+-------+-------+-------+









# class VLF(nn.Module):
#     def __init__(self,
#                  img_size=512,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionBlock(out_channels, l_dim, num_heads, dropout,
#                                 img_size // (2 ** (i + 2))) for i in range(len(in_channels))
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout, window, mlp_ratio=4.):
#         super(CrossAttentionBlock, self).__init__()
#         self.norm1_vis = nn.LayerNorm(dim)
#         self.norm1_lang = nn.LayerNorm(l_dim)
#         self.cross_att = CrossAttentionModule(dim, l_dim, num_heads, dropout)
#
#         self.norm2_mm = nn.LayerNorm(dim)
#         self.self_att = SelfAttentionModule(dim, num_heads, dropout=dropout, window=window)
#
#         self.norm3_emm = nn.LayerNorm(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
#
#     def forward(self, x, l, l_mask):
#         x = x.flatten(2).transpose(1, 2)
#         x_res = x.clone()
#
#         mm = x_res + self.cross_att(self.norm1_vis(x), self.norm1_lang(l.permute(0, 2, 1)), l_mask)
#         emm = mm + self.self_att(self.norm2_mm(mm))
#         out = emm + self.mlp(self.norm3_emm(emm))
#
#         return out
#
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         x_res = x.clone()
#         B, N, C = x.shape
#
#         q = self.q_vis(x)
#         k = self.k_lang(l)
#         v = self.v_lang(l)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         out = x_res * mm
#
#         return out
#
#
# class SelfAttentionModule(nn.Module):
#     def __init__(self, dim, num_heads, qkv_bias=True, dropout=0., agent_num=49, window=16):
#         super(SelfAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.agent_num = agent_num
#         self.window = window
#
#         self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
#
#         self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
#         self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
#         self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
#         self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
#         trunc_normal_(self.an_bias, std=.02)
#         trunc_normal_(self.na_bias, std=.02)
#         trunc_normal_(self.ah_bias, std=.02)
#         trunc_normal_(self.aw_bias, std=.02)
#         trunc_normal_(self.ha_bias, std=.02)
#         trunc_normal_(self.wa_bias, std=.02)
#
#         pool_size = int(agent_num ** 0.5)
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
#
#     def forward(self, x):
#         x_res = x.clone()
#
#         B, N, C = x.shape
#         H = W = int(math.sqrt(N))
#
#         qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         q, k, v = qkv.unbind(0)
#
#         agent_tokens = self.pool(q.reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         agent_tokens = agent_tokens.reshape(B, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
#         position_bias1 = position_bias1.reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias = position_bias1 + position_bias2
#
#         agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
#         agent_attn = self.attn_drop(agent_attn)
#         agent_v = agent_attn @ v
#
#         agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
#         agent_bias1 = agent_bias1.reshape(1, self.num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(B, 1, 1, 1)
#         agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, self.num_heads, -1, self.agent_num).repeat(B, 1, 1, 1)
#         agent_bias = agent_bias1 + agent_bias2
#
#         q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
#         q_attn = self.attn_drop(q_attn)
#         x = q_attn @ agent_v
#
#         x = x.transpose(1, 2).reshape(B, N, C)
#         v = v.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
#         x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(B, N, C)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         out = x_res * x
#
#         return out
#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
# V17 Results
# +-------+-------+-------+-------+-------+------+------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 | oIoU | mIoU |
# +-------+-------+-------+-------+-------+------+------+
# | 80.71 | 72.14 | 55.99 | 29.12 |  8.3  | 81.1 | 65.7 |
# +-------+-------+-------+-------+-------+------+------+
# +-------+-------+-------+-------+-------+------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 | oIoU |  mIoU |
# +-------+-------+-------+-------+-------+------+-------+
# | 81.37 | 72.25 | 56.37 | 29.18 |  8.24 | 80.8 | 65.83 |
# +-------+-------+-------+-------+-------+------+-------+








# class VLF(nn.Module):
#     def __init__(self,
#                  img_size=512,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionBlock(out_channels, l_dim, num_heads, dropout,
#                                 img_size // (2 ** (i + 2))) for i in range(len(in_channels))
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout, window, mlp_ratio=4.):
#         super(CrossAttentionBlock, self).__init__()
#         self.norm1_vis = nn.LayerNorm(dim)
#         self.norm1_lang = nn.LayerNorm(l_dim)
#         self.cross_att = CrossAttentionModule(dim, l_dim, num_heads, dropout)
#
#         self.norm2_mm = nn.LayerNorm(dim)
#         self.norm2_lang = nn.LayerNorm(l_dim)
#         self.self_att = SelfAttentionModule(dim, l_dim, num_heads, dropout=dropout, window=window)
#
#         self.norm3_emm = nn.LayerNorm(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
#
#     def forward(self, x, l, l_mask):
#         x = x.flatten(2).transpose(1, 2)
#         x_res = x.clone()
#
#         mm = x_res + self.cross_att(self.norm1_vis(x), self.norm1_lang(l.permute(0, 2, 1)), l_mask)
#         emm = mm + self.self_att(self.norm2_mm(mm), self.norm2_lang(l.permute(0, 2, 1)))
#         out = emm + self.mlp(self.norm3_emm(emm))
#
#         return out
#
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         x_res = x.clone()
#         B, N, C = x.shape
#
#         q = self.q_vis(x)
#         k = self.k_lang(l)
#         v = self.v_lang(l)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         out = x_res * mm
#
#         return out
#
#
# class SelfAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, qkv_bias=True, dropout=0., agent_num=20, window=16):
#         super(SelfAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.lang_proj = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.agent_num = agent_num
#         self.window = window
#
#         self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
#
#         self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
#         self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
#         self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
#         self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
#         trunc_normal_(self.an_bias, std=.02)
#         trunc_normal_(self.na_bias, std=.02)
#         trunc_normal_(self.ah_bias, std=.02)
#         trunc_normal_(self.aw_bias, std=.02)
#         trunc_normal_(self.ha_bias, std=.02)
#         trunc_normal_(self.wa_bias, std=.02)
#
#     def forward(self, x, l):
#         x_res = x.clone()
#
#         B, N, C = x.shape
#         H = W = int(math.sqrt(N))
#
#         qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         q, k, v = qkv.unbind(0)
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         language_agent_tokens = self.lang_proj(l)
#         language_agent_tokens = language_agent_tokens.reshape(B, self.agent_num,
#                                                               self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
#         position_bias1 = position_bias1.reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias = position_bias1 + position_bias2
#
#         agent_attn = self.softmax((language_agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
#         agent_attn = self.attn_drop(agent_attn)
#         agent_v = agent_attn @ v
#
#         agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
#         agent_bias1 = agent_bias1.reshape(1, self.num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(B, 1, 1, 1)
#         agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, self.num_heads, -1, self.agent_num).repeat(B, 1, 1, 1)
#         agent_bias = agent_bias1 + agent_bias2
#
#         q_attn = self.softmax((q * self.scale) @ language_agent_tokens.transpose(-2, -1) + agent_bias)
#         q_attn = self.attn_drop(q_attn)
#         x = q_attn @ agent_v
#
#         x = x.transpose(1, 2).reshape(B, N, C)
#         v = v.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
#         x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(B, N, C)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         out = x_res * x
#
#         return out
#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
# V18 Results
# +-------+-------+-------+-------+-------+------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 | oIoU |  mIoU |
# +-------+-------+-------+-------+-------+------+-------+
# |  80.0 | 71.26 | 54.62 | 26.04 |  7.69 | 80.5 | 65.06 |
# +-------+-------+-------+-------+-------+------+-------+




# class VLF(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionBlock(out_channels, l_dim, num_heads, dropout) for _ in in_channels
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout, mlp_ratio=4.):
#         super(CrossAttentionBlock, self).__init__()
#         self.norm1_vis = nn.LayerNorm(dim)
#         self.norm1_lang = nn.LayerNorm(l_dim)
#         self.cross_att = CrossAttentionModule(dim, l_dim, num_heads, dropout)
#
#         self.norm2_mm = nn.LayerNorm(dim)
#         self.norm2_lang = nn.LayerNorm(l_dim)
#         self.self_att = SelfAttentionModule(dim, l_dim, num_heads, dropout=dropout)
#
#         self.norm3_emm = nn.LayerNorm(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
#
#     def forward(self, x, l, l_mask):
#         x = x.flatten(2).transpose(1, 2)
#         x_res = x.clone()
#
#         mm = x_res + self.cross_att(self.norm1_vis(x), self.norm1_lang(l.permute(0, 2, 1)), l_mask)
#         emm = mm + self.self_att(self.norm2_mm(mm), self.norm2_lang(l.permute(0, 2, 1)))
#         out = emm + self.mlp(self.norm3_emm(emm))
#
#         return out
#
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         x_res = x.clone()
#         B, N, C = x.shape
#
#         q = self.q_vis(x)
#         k = self.k_lang(l)
#         v = self.v_lang(l)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         out = x_res * mm
#
#         return out
#
#
# class SelfAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, qkv_bias=True, dropout=0., agent_num=20):
#         super(SelfAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.lang_proj = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.agent_num = agent_num
#
#         self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
#
#
#     def forward(self, x, l):
#         x_res = x.clone()
#
#         B, N, C = x.shape
#         H = W = int(math.sqrt(N))
#
#         qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         q, k, v = qkv.unbind(0)
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         language_agent_tokens = self.lang_proj(l)
#         language_agent_tokens = language_agent_tokens.reshape(B, self.agent_num,
#                                                               self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         agent_attn = self.softmax((language_agent_tokens * self.scale) @ k.transpose(-2, -1))
#         agent_attn = self.attn_drop(agent_attn)
#         agent_v = agent_attn @ v
#
#         q_attn = self.softmax((q * self.scale) @ language_agent_tokens.transpose(-2, -1))
#         q_attn = self.attn_drop(q_attn)
#         x = q_attn @ agent_v
#
#         x = x.transpose(1, 2).reshape(B, N, C)
#         v = v.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
#         x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(B, N, C)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         out = x_res * x
#
#         return out
#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
# V19 Results
# +-------+-------+-------+-------+-------+------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 | oIoU |  mIoU |
# +-------+-------+-------+-------+-------+------+-------+
# | 80.88 | 71.54 | 55.66 | 28.19 |  7.75 | 80.8 | 65.55 |
# +-------+-------+-------+-------+-------+------+-------+





# class VLF(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionBlock(out_channels, l_dim, num_heads, dropout) for _ in in_channels
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout, mlp_ratio=4.):
#         super(CrossAttentionBlock, self).__init__()
#         self.norm1_vis = nn.LayerNorm(dim)
#         self.norm1_lang = nn.LayerNorm(l_dim)
#         self.cross_att = CrossAttentionModule(dim, l_dim, num_heads, dropout)
#
#         self.norm2_mm = nn.LayerNorm(dim)
#         self.norm2_lang = nn.LayerNorm(l_dim)
#         self.self_att = SelfAttentionModule(dim, l_dim, num_heads, dropout=dropout)
#
#         self.norm3_emm = nn.LayerNorm(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
#
#     def forward(self, x, l, l_mask):
#         x = x.flatten(2).transpose(1, 2)
#         x_res = x.clone()
#
#         mm = x_res + self.cross_att(self.norm1_vis(x), self.norm1_lang(l.permute(0, 2, 1)), l_mask)
#         emm = mm + self.self_att(self.norm2_mm(mm), self.norm2_lang(l.permute(0, 2, 1)))
#         out = emm + self.mlp(self.norm3_emm(emm))
#
#         return out
#
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         x_res = x.clone()
#         B, N, C = x.shape
#
#         q = self.q_vis(x)
#         k = self.k_lang(l)
#         v = self.v_lang(l)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         out = x_res * mm
#
#         return out
#
#
# class SelfAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, qkv_bias=True, dropout=0., agent_num=20):
#         super(SelfAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.lang_proj = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.agent_num = agent_num
#
#     def forward(self, x, l):
#         x_res = x.clone()
#
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         q, k, v = qkv.unbind(0)
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         language_agent_tokens = self.lang_proj(l)
#         language_agent_tokens = language_agent_tokens.reshape(B, self.agent_num,
#                                                               self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         agent_attn = self.softmax((language_agent_tokens * self.scale) @ k.transpose(-2, -1))
#         agent_attn = self.attn_drop(agent_attn)
#         agent_v = agent_attn @ v
#
#         q_attn = self.softmax((q * self.scale) @ language_agent_tokens.transpose(-2, -1))
#         q_attn = self.attn_drop(q_attn)
#         x = q_attn @ agent_v
#
#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         out = x_res * x
#
#         return out
#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
# V20 Results
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 80.33 | 71.54 | 55.33 | 26.32 |  7.69 | 80.49 | 65.51 |
# +-------+-------+-------+-------+-------+-------+-------+






# class VLF(nn.Module):
#     def __init__(self,
#                  img_size=512,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionBlock(out_channels, l_dim, num_heads, dropout,
#                                 img_size // (2 ** (i + 2))) for i in range(len(in_channels))
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout, window, mlp_ratio=4.):
#         super(CrossAttentionBlock, self).__init__()
#         self.norm1_vis = nn.LayerNorm(dim)
#         self.norm1_lang = nn.LayerNorm(l_dim)
#         self.cross_att = CrossAttentionModule(dim, l_dim, num_heads, dropout)
#
#         self.norm2_mm = nn.LayerNorm(dim)
#         self.self_att = SelfAttentionModule(dim, num_heads, dropout=dropout, window=window)
#
#         self.norm3_emm = nn.LayerNorm(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
#
#     def forward(self, x, l, l_mask):
#         x = x.flatten(2).transpose(1, 2)
#         x_res = x.clone()
#
#         mm = x_res + self.cross_att(self.norm1_vis(x), self.norm1_lang(l.permute(0, 2, 1)), l_mask)
#         emm = mm + self.self_att(self.norm2_mm(mm))
#         out = emm + self.mlp(self.norm3_emm(emm))
#
#         return out
#
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         B, N, C = x.shape
#
#         q = self.q_vis(x)
#         k = self.k_lang(l)
#         v = self.v_lang(l)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         return mm
#
#
# class SelfAttentionModule(nn.Module):
#     def __init__(self, dim, num_heads, qkv_bias=True, dropout=0., agent_num=49, window=16):
#         super(SelfAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.agent_num = agent_num
#         self.window = window
#
#         self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
#
#         self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
#         self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
#         self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
#         self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
#         trunc_normal_(self.an_bias, std=.02)
#         trunc_normal_(self.na_bias, std=.02)
#         trunc_normal_(self.ah_bias, std=.02)
#         trunc_normal_(self.aw_bias, std=.02)
#         trunc_normal_(self.ha_bias, std=.02)
#         trunc_normal_(self.wa_bias, std=.02)
#
#         pool_size = int(agent_num ** 0.5)
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
#
#     def forward(self, x):
#         B, N, C = x.shape
#         H = W = int(math.sqrt(N))
#
#         qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         q, k, v = qkv.unbind(0)
#
#         agent_tokens = self.pool(q.reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         agent_tokens = agent_tokens.reshape(B, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
#         position_bias1 = position_bias1.reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias = position_bias1 + position_bias2
#
#         agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
#         agent_attn = self.attn_drop(agent_attn)
#         agent_v = agent_attn @ v
#
#         agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
#         agent_bias1 = agent_bias1.reshape(1, self.num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(B, 1, 1, 1)
#         agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, self.num_heads, -1, self.agent_num).repeat(B, 1, 1, 1)
#         agent_bias = agent_bias1 + agent_bias2
#
#         q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
#         q_attn = self.attn_drop(q_attn)
#         x = q_attn @ agent_v
#
#         x = x.transpose(1, 2).reshape(B, N, C)
#         v = v.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
#         x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(B, N, C)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x
#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
# V21 Results
# +-------+-------+-------+-------+-------+------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 | oIoU |  mIoU |
# +-------+-------+-------+-------+-------+------+-------+
# | 78.52 | 68.08 | 50.05 |  21.1 |  6.87 | 80.2 | 63.39 |
# +-------+-------+-------+-------+-------+------+-------+








# class VLF(nn.Module):
#     def __init__(self,
#                  img_size=512,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionBlock(out_channels, l_dim, num_heads, dropout,
#                                 img_size // (2 ** (i + 2))) for i in range(len(in_channels))
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout, window, mlp_ratio=4.):
#         super(CrossAttentionBlock, self).__init__()
#         self.norm1_vis = nn.LayerNorm(dim)
#         self.norm1_lang = nn.LayerNorm(l_dim)
#         self.cross_att = CrossAttentionModule(dim, l_dim, num_heads, dropout)
#
#         self.norm2_mm = nn.LayerNorm(dim)
#         self.self_att = SelfAttentionModule(dim, num_heads, dropout=dropout, window=window)
#
#         self.norm3_emm = nn.LayerNorm(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
#
#     def forward(self, x, l, l_mask):
#         x = x.flatten(2).transpose(1, 2)
#         x_res = x.clone()
#
#         mm = x_res * self.cross_att(self.norm1_vis(x), self.norm1_lang(l.permute(0, 2, 1)), l_mask)
#         emm = mm * self.self_att(self.norm2_mm(mm))
#         out = emm * self.mlp(self.norm3_emm(emm))
#
#         return out
#
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         x_res = x.clone()
#         B, N, C = x.shape
#
#         q = self.q_vis(x)
#         k = self.k_lang(l)
#         v = self.v_lang(l)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         out = x_res * mm
#
#         return out
#
#
# class SelfAttentionModule(nn.Module):
#     def __init__(self, dim, num_heads, qkv_bias=True, dropout=0., agent_num=49, window=16):
#         super(SelfAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.agent_num = agent_num
#         self.window = window
#
#         self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
#
#         self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
#         self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
#         self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
#         self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
#         trunc_normal_(self.an_bias, std=.02)
#         trunc_normal_(self.na_bias, std=.02)
#         trunc_normal_(self.ah_bias, std=.02)
#         trunc_normal_(self.aw_bias, std=.02)
#         trunc_normal_(self.ha_bias, std=.02)
#         trunc_normal_(self.wa_bias, std=.02)
#
#         pool_size = int(agent_num ** 0.5)
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
#
#     def forward(self, x):
#         x_res = x.clone()
#
#         B, N, C = x.shape
#         H = W = int(math.sqrt(N))
#
#         qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         q, k, v = qkv.unbind(0)
#
#         agent_tokens = self.pool(q.reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         agent_tokens = agent_tokens.reshape(B, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
#         position_bias1 = position_bias1.reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias = position_bias1 + position_bias2
#
#         agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
#         agent_attn = self.attn_drop(agent_attn)
#         agent_v = agent_attn @ v
#
#         agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
#         agent_bias1 = agent_bias1.reshape(1, self.num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(B, 1, 1, 1)
#         agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, self.num_heads, -1, self.agent_num).repeat(B, 1, 1, 1)
#         agent_bias = agent_bias1 + agent_bias2
#
#         q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
#         q_attn = self.attn_drop(q_attn)
#         x = q_attn @ agent_v
#
#         x = x.transpose(1, 2).reshape(B, N, C)
#         v = v.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
#         x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(B, N, C)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         out = x_res * x
#
#         return out
#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
# V22 Results
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 72.09 | 59.12 | 36.26 | 14.62 |  3.96 | 76.27 | 58.56 |
# +-------+-------+-------+-------+-------+-------+-------+







# class VLF(nn.Module):
#     def __init__(self,
#                  img_size=512,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionBlock(out_channels, l_dim, num_heads, dropout,
#                                 img_size // (2 ** (i + 2))) for i in range(len(in_channels))
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout, window, mlp_ratio=4.):
#         super(CrossAttentionBlock, self).__init__()
#         self.cross_att = CrossAttentionModule(dim, l_dim, num_heads, dropout)
#         self.norm1 = nn.LayerNorm(dim)
#
#         self.self_att = SelfAttentionModule(dim, num_heads, dropout=dropout, window=window)
#         self.norm2 = nn.LayerNorm(dim)
#
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
#         self.norm3 = nn.LayerNorm(dim)
#
#     def forward(self, x, l, l_mask):
#         x = x.flatten(2).transpose(1, 2)
#         x_res = x.clone()
#
#         mm = self.norm1(x_res + self.cross_att(x, l.permute(0, 2, 1), l_mask))
#         emm = self.norm2(mm + self.self_att(mm))
#         out = self.norm3(emm + self.mlp(emm))
#
#         return out
#
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         x_res = x.clone()
#         B, N, C = x.shape
#
#         q = self.q_vis(x)
#         k = self.k_lang(l)
#         v = self.v_lang(l)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         out = x_res * mm
#
#         return out
#
#
# class SelfAttentionModule(nn.Module):
#     def __init__(self, dim, num_heads, qkv_bias=True, dropout=0., agent_num=49, window=16):
#         super(SelfAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.agent_num = agent_num
#         self.window = window
#
#         self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
#
#         self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
#         self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
#         self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
#         self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
#         trunc_normal_(self.an_bias, std=.02)
#         trunc_normal_(self.na_bias, std=.02)
#         trunc_normal_(self.ah_bias, std=.02)
#         trunc_normal_(self.aw_bias, std=.02)
#         trunc_normal_(self.ha_bias, std=.02)
#         trunc_normal_(self.wa_bias, std=.02)
#
#         pool_size = int(agent_num ** 0.5)
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
#
#     def forward(self, x):
#         x_res = x.clone()
#
#         B, N, C = x.shape
#         H = W = int(math.sqrt(N))
#
#         qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         q, k, v = qkv.unbind(0)
#
#         agent_tokens = self.pool(q.reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         agent_tokens = agent_tokens.reshape(B, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
#         position_bias1 = position_bias1.reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias = position_bias1 + position_bias2
#
#         agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
#         agent_attn = self.attn_drop(agent_attn)
#         agent_v = agent_attn @ v
#
#         agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
#         agent_bias1 = agent_bias1.reshape(1, self.num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(B, 1, 1, 1)
#         agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, self.num_heads, -1, self.agent_num).repeat(B, 1, 1, 1)
#         agent_bias = agent_bias1 + agent_bias2
#
#         q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
#         q_attn = self.attn_drop(q_attn)
#         x = q_attn @ agent_v
#
#         x = x.transpose(1, 2).reshape(B, N, C)
#         v = v.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
#         x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(B, N, C)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         out = x_res * x
#
#         return out
#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
# V23 Results
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 77.09 | 67.97 | 48.85 | 22.31 |  7.25 | 80.13 | 63.29 |
# +-------+-------+-------+-------+-------+-------+-------+








# class VLF(nn.Module):
#     def __init__(self,
#                  img_size=512,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(dim, out_channels) for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionBlock(out_channels, l_dim, num_heads, dropout,
#                                 img_size // (2 ** (i + 2))) for i in range(len(in_channels))
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout, window, mlp_ratio=4.):
#         super(CrossAttentionBlock, self).__init__()
#         self.norm1_vis = nn.LayerNorm(dim)
#         self.norm1_lang = nn.LayerNorm(l_dim)
#         self.cross_att = CrossAttentionModule(dim, l_dim, num_heads, dropout)
#
#         self.norm2_mm = nn.LayerNorm(dim)
#         self.self_att = SelfAttentionModule(dim, num_heads, dropout=dropout, window=window)
#
#         self.norm3_emm = nn.LayerNorm(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
#
#     def forward(self, x, l, l_mask):
#         x = x.flatten(2).transpose(1, 2)
#         x_res = x.clone()
#
#         mm = x_res + self.cross_att(self.norm1_vis(x), self.norm1_lang(l.permute(0, 2, 1)), l_mask)
#         emm = mm + self.self_att(self.norm2_mm(mm))
#         out = emm + self.mlp(self.norm3_emm(emm))
#
#         return out
#
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         x_res = x.clone()
#         B, N, C = x.shape
#
#         l_mask = l_mask.permute(0, 2, 1)
#
#         q = self.q_vis(x)
#         k = (self.k_lang(l).permute(0, 2, 1) * l_mask).permute(0, 2, 1)
#         v = (self.v_lang(l).permute(0, 2, 1) * l_mask).permute(0, 2, 1)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         l_mask = l_mask.unsqueeze(1)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale + (1e4 * l_mask - 1e4)
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         out = x_res * mm
#
#         return out
#
#
# class SelfAttentionModule(nn.Module):
#     def __init__(self, dim, num_heads, qkv_bias=True, dropout=0., agent_num=49, window=16):
#         super(SelfAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.agent_num = agent_num
#         self.window = window
#
#         self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
#
#         self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
#         self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
#         self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
#         self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
#         trunc_normal_(self.an_bias, std=.02)
#         trunc_normal_(self.na_bias, std=.02)
#         trunc_normal_(self.ah_bias, std=.02)
#         trunc_normal_(self.aw_bias, std=.02)
#         trunc_normal_(self.ha_bias, std=.02)
#         trunc_normal_(self.wa_bias, std=.02)
#
#         pool_size = int(agent_num ** 0.5)
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
#
#     def forward(self, x):
#         x_res = x.clone()
#
#         B, N, C = x.shape
#         H = W = int(math.sqrt(N))
#
#         qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         q, k, v = qkv.unbind(0)
#
#         agent_tokens = self.pool(q.reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         agent_tokens = agent_tokens.reshape(B, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
#         position_bias1 = position_bias1.reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias = position_bias1 + position_bias2
#
#         agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
#         agent_attn = self.attn_drop(agent_attn)
#         agent_v = agent_attn @ v
#
#         agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
#         agent_bias1 = agent_bias1.reshape(1, self.num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(B, 1, 1, 1)
#         agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, self.num_heads, -1, self.agent_num).repeat(B, 1, 1, 1)
#         agent_bias = agent_bias1 + agent_bias2
#
#         q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
#         q_attn = self.attn_drop(q_attn)
#         x = q_attn @ agent_v
#
#         x = x.transpose(1, 2).reshape(B, N, C)
#         v = v.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
#         x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(B, N, C)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         out = x_res * x
#
#         return out
#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
# V24 Results
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 80.05 | 71.21 | 55.05 |  28.9 |  8.08 | 80.89 | 65.15 |
# +-------+-------+-------+-------+-------+-------+-------+




# class VLF(nn.Module):
#     def __init__(self,
#                  img_size=512,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.):
#         super(VLF, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             nn.Sequential(nn.Linear(dim, out_channels),
#                           nn.GELU(),
#                           nn.Dropout(dropout))
#             for dim in in_channels
#         ])
#
#         self.cross_att_layers = nn.ModuleList([
#             CrossAttentionBlock(out_channels, l_dim, num_heads, dropout,
#                                 img_size // (2 ** (i + 2))) for i in range(len(in_channels))
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for idx, x in enumerate(inputs):
#             x = self.vis_proj_layers[idx](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#             B, C, H, W = x.shape
#             x = self.cross_att_layers[idx](x, l, l_mask)
#             out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout, window, mlp_ratio=4.):
#         super(CrossAttentionBlock, self).__init__()
#         self.norm1_vis = nn.LayerNorm(dim)
#         self.norm1_lang = nn.LayerNorm(l_dim)
#         self.cross_att = CrossAttentionModule(dim, l_dim, num_heads, dropout)
#
#         self.norm2_mm = nn.LayerNorm(dim)
#         self.self_att = SelfAttentionModule(dim, num_heads, dropout=dropout, window=window)
#
#         self.norm3_emm = nn.LayerNorm(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
#
#     def forward(self, x, l, l_mask):
#         x = x.flatten(2).transpose(1, 2)
#         x_res = x.clone()
#
#         mm = x_res + self.cross_att(self.norm1_vis(x), self.norm1_lang(l.permute(0, 2, 1)), l_mask)
#         emm = mm + self.self_att(self.norm2_mm(mm))
#         out = emm + self.mlp(self.norm3_emm(emm))
#
#         return out
#
#
# class CrossAttentionModule(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CrossAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_vis = nn.Linear(dim, dim)
#         self.k_lang = nn.Linear(l_dim, dim)
#         self.v_lang = nn.Linear(l_dim, dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_mm = nn.Linear(dim, dim)
#         self.proj_mm_drop = nn.Dropout(dropout)
#
#     def forward(self, x, l, l_mask):
#         x_res = x.clone()
#         B, N, C = x.shape
#
#         q = self.q_vis(x)
#         k = self.k_lang(l)
#         v = self.v_lang(l)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         mm = self.proj_mm(mm)
#         mm = self.proj_mm_drop(mm)
#
#         out = x_res * mm
#
#         return out
#
#
# class SelfAttentionModule(nn.Module):
#     def __init__(self, dim, num_heads, qkv_bias=True, dropout=0., agent_num=49, window=16):
#         super(SelfAttentionModule, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.agent_num = agent_num
#         self.window = window
#
#         self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
#
#         self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
#         self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
#         self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
#         self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
#         trunc_normal_(self.an_bias, std=.02)
#         trunc_normal_(self.na_bias, std=.02)
#         trunc_normal_(self.ah_bias, std=.02)
#         trunc_normal_(self.aw_bias, std=.02)
#         trunc_normal_(self.ha_bias, std=.02)
#         trunc_normal_(self.wa_bias, std=.02)
#
#         pool_size = int(agent_num ** 0.5)
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
#
#     def forward(self, x):
#         x_res = x.clone()
#
#         B, N, C = x.shape
#         H = W = int(math.sqrt(N))
#
#         qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         q, k, v = qkv.unbind(0)
#
#         agent_tokens = self.pool(q.reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         agent_tokens = agent_tokens.reshape(B, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
#         position_bias1 = position_bias1.reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
#         position_bias = position_bias1 + position_bias2
#
#         agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
#         agent_attn = self.attn_drop(agent_attn)
#         agent_v = agent_attn @ v
#
#         agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
#         agent_bias1 = agent_bias1.reshape(1, self.num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(B, 1, 1, 1)
#         agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, self.num_heads, -1, self.agent_num).repeat(B, 1, 1, 1)
#         agent_bias = agent_bias1 + agent_bias2
#
#         q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
#         q_attn = self.attn_drop(q_attn)
#         x = q_attn @ agent_v
#
#         x = x.transpose(1, 2).reshape(B, N, C)
#         v = v.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
#         x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(B, N, C)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         out = x_res * x
#
#         return out
#
#
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
# V25 Results
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 80.11 | 70.22 | 52.03 | 21.87 |  6.7  | 79.92 | 64.22 |
# +-------+-------+-------+-------+-------+-------+-------+







class VLF(nn.Module):
    def __init__(self,
                 img_size=512,
                 in_channels=[96, 192, 384, 768],
                 out_channels=256,
                 l_dim=768,
                 num_heads=8,
                 dropout=0.):
        super(VLF, self).__init__()

        self.vis_proj_layers = nn.ModuleList([
            CBR(dim, out_channels) for dim in in_channels
        ])

        self.cross_att_layers = nn.ModuleList([
            CrossAttentionBlock(out_channels, l_dim, num_heads, dropout,
                                img_size // (2 ** (i + 2))) for i in range(len(in_channels))
        ])

    def forward(self, inputs, l, l_mask):
        outs = []
        for idx, x in enumerate(inputs):
            x = self.vis_proj_layers[idx](x)
            B, C, H, W = x.shape
            x = self.cross_att_layers[idx](x, l, l_mask)
            out = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(out)

        return outs


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, l_dim, num_heads, dropout, window, mlp_ratio=4.):
        super(CrossAttentionBlock, self).__init__()
        self.norm1_vis = nn.LayerNorm(dim)
        self.norm1_lang = nn.LayerNorm(l_dim)
        self.cross_att = CrossAttentionModule(dim, l_dim, num_heads, dropout)

        self.norm2_mm = nn.LayerNorm(dim)
        self.self_att = SelfAttentionModule(dim, num_heads, dropout=dropout, window=window)

        self.norm3_emm = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, l, l_mask):
        x = x.flatten(2).transpose(1, 2)
        x_res = x.clone()

        mm = x_res + self.cross_att(self.norm1_vis(x), self.norm1_lang(l.permute(0, 2, 1)), l_mask)
        emm = mm + self.self_att(self.norm2_mm(mm))
        out = emm + self.mlp(self.norm3_emm(emm))

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

    def forward(self, x, l, l_mask):
        x_res = x.clone()
        B, N, C = x.shape

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
    def __init__(self, dim, num_heads, qkv_bias=True, dropout=0., agent_num=49, window=16):
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

        B, N, C = x.shape
        H = W = int(math.sqrt(N))

        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)

        agent_tokens = self.pool(q.reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(B, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
        position_bias1 = position_bias1.reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
        position_bias = position_bias1 + position_bias2

        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, self.num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(B, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, self.num_heads, -1, self.agent_num).repeat(B, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2

        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(B, N, C)
        v = v.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        out = x_res * x

        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x