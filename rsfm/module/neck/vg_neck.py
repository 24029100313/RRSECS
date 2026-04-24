from torch import nn
from rsfm.module.neck.utils import CBR




# class vg_neck(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.,
#                  **kwargs):
#         super(vg_neck, self).__init__()
#
#         self.fusion = nn.ModuleList([
#             CAM(in_channels[i], l_dim, num_heads, dropout) for i in range(len(in_channels))
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#
#         for layer, x in zip(self.fusion, inputs):
#             B, C, H, W = x.shape
#             mm = layer(x.flatten(2).transpose(1, 2), l, l_mask)
#             out = mm.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs


# v1
# class CAM(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CAM, self).__init__()
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

# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 77.26 | 71.45 | 62.66 |  48.4 | 25.36 | 76.81 | 66.35 |
# +-------+-------+-------+-------+-------+-------+-------+


# v2
# class CAM(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CAM, self).__init__()
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
#         x_res = x.clone()
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
#         out = mm * x_res
#
#         return out
#
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 80.73 | 76.51 | 68.43 | 56.13 | 33.35 | 79.36 | 69.98 |
# +-------+-------+-------+-------+-------+-------+-------+


# v3
# class CAM(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CAM, self).__init__()
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
#         x_res = x.clone()
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
#         out = mm + x_res
#
#         return out
#
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 79.75 | 75.71 | 68.83 | 55.99 | 34.55 | 77.82 | 69.11 |
# +-------+-------+-------+-------+-------+-------+-------+


# v4
# class vg_neck(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.,
#                  **kwargs):
#         super(vg_neck, self).__init__()
#
#         self.vis_proj = nn.ModuleList([
#             CBR(in_channels[i], in_channels[i], 1) for i in range(len(in_channels))
#         ])
#
#         self.fusion = nn.ModuleList([
#             CAM(in_channels[i], l_dim, num_heads, dropout) for i in range(len(in_channels))
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#
#         for v_proj, layer, x in zip(self.vis_proj, self.fusion, inputs):
#             x = v_proj(x)
#             B, C, H, W = x.shape
#             mm = layer(x.flatten(2).transpose(1, 2), l, l_mask)
#             out = mm.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 79.71 | 75.13 | 67.67 |  55.2 | 32.99 | 78.76 | 68.93 |
# +-------+-------+-------+-------+-------+-------+-------+


# v5
# class vg_neck(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.,
#                  **kwargs):
#         super(vg_neck, self).__init__()
#
#         self.vis_proj = nn.ModuleList([
#             CBR(in_channels[i], out_channels, 1) for i in range(len(in_channels))
#         ])
#
#         self.fusion = nn.ModuleList([
#             CAM(out_channels, l_dim, num_heads, dropout) for i in range(len(in_channels))
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#
#         for v_proj, layer, x in zip(self.vis_proj, self.fusion, inputs):
#             x = v_proj(x)
#             B, C, H, W = x.shape
#             mm = layer(x.flatten(2).transpose(1, 2), l, l_mask)
#             out = mm.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
# +-------+-------+-------+-------+-------+------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 | oIoU |  mIoU |
# +-------+-------+-------+-------+-------+------+-------+
# | 79.57 | 75.58 | 67.72 | 54.44 | 32.42 | 78.5 | 69.03 |
# +-------+-------+-------+-------+-------+------+-------+


# v6
# class CAM(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CAM, self).__init__()
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
#         x_res = x.clone()
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
#         out = mm * x_res
#
#         return out


# class vg_neck(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.,
#                  **kwargs):
#         super(vg_neck, self).__init__()
#
#         self.vis_proj = nn.ModuleList([
#             CBR(in_channels[i], out_channels, 1) for i in range(len(in_channels))
#         ])
#
#         self.lang_proj = nn.ModuleList([
#             nn.Sequential(nn.Linear(l_dim, out_channels),
#                           nn.GELU(),
#                           nn.Dropout(dropout)) for _ in in_channels
#         ])
#
#         self.fusion = nn.ModuleList([
#             CAM(out_channels, out_channels, num_heads, dropout) for i in range(len(in_channels))
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#         for v_proj, l_proj, layer, x in zip(self.vis_proj, self.lang_proj, self.fusion, inputs):
#             x = v_proj(x)
#             lp = l_proj(l)
#             B, C, H, W = x.shape
#             mm = layer(x.flatten(2).transpose(1, 2), lp, l_mask)
#             out = mm.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs

# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 79.13 | 75.04 | 68.25 | 55.11 |  33.3 | 78.82 | 68.79 |
# +-------+-------+-------+-------+-------+-------+-------+



# v7
# class SAM(nn.Module):
#     def __init__(self, q_dim, k_dim, v_dim, num_heads, dropout, hidden_dim=None):
#         super(SAM, self).__init__()
#
#         self.hidden_dim = hidden_dim if hidden_dim is not None else q_dim
#         self.num_heads = num_heads
#         self.head_dim = self.hidden_dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q = nn.Linear(q_dim, self.hidden_dim)
#         self.k = nn.Linear(k_dim, self.hidden_dim)
#         self.v = nn.Linear(v_dim, self.hidden_dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.proj_drop = nn.Dropout(dropout)
#
#     def forward(self, query, key, value, with_star_residual=True):
#         B, N, C = query.shape
#
#         if with_star_residual:
#             query_res = query.clone()
#
#         q = self.q(query)
#         k = self.k(key)
#         v = self.v(value)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         out = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         out = self.proj(out)
#         out = self.proj_drop(out)
#
#         if with_star_residual:
#             out = out * query_res
#
#         return out
#
#
# class vg_neck(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.,
#                  **kwargs):
#         super(vg_neck, self).__init__()
#
#         self.cascade_lang_vis_fusion_layers = nn.ModuleList([
#             SAM(l_dim, in_channels[i], in_channels[i], num_heads, dropout) for i in range(len(in_channels))
#         ])
#
#         self.hierarchical_vis_lang_fusion_layers = nn.ModuleList([
#             SAM(in_channels[i], l_dim, l_dim, num_heads, dropout) for i in range(len(in_channels))
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#
#         # cascade language-vision fusion
#         for x, clvf_layer in zip(inputs, self.cascade_lang_vis_fusion_layers):
#             x = x.flatten(2).transpose(1, 2)
#             l = clvf_layer(l, x, x)
#
#         # hierarchical vision-language fusion
#         for x, hvlf_layer in zip(inputs, self.hierarchical_vis_lang_fusion_layers):
#             B, C, H, W = x.shape
#             x = x.flatten(2).transpose(1, 2)
#             out = hvlf_layer(x, l, l)
#             out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 78.46 |  74.2 | 67.45 | 55.46 | 33.93 | 77.06 | 68.19 |
# +-------+-------+-------+-------+-------+-------+-------+



# v8
# class vg_neck(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.,
#                  **kwargs):
#         super(vg_neck, self).__init__()
#
#         self.out_channels = out_channels
#
#         self.fusion = nn.ModuleList([
#             CAM(in_channels[i], l_dim, l_dim, num_heads, dropout, out_channels) for i in range(len(in_channels))
#         ])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#
#         for layer, x in zip(self.fusion, inputs):
#             B, _, H, W = x.shape
#             mm = layer(x.flatten(2).transpose(1, 2), l, l)
#             out = mm.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
# class CAM(nn.Module):
#     def __init__(self, q_dim, k_dim, v_dim, num_heads, dropout, hidden_dim=None):
#         super(CAM, self).__init__()
#
#         self.hidden_dim = hidden_dim if hidden_dim is not None else q_dim
#         self.num_heads = num_heads
#         self.head_dim = self.hidden_dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q = nn.Linear(q_dim, self.hidden_dim)
#         self.k = nn.Linear(k_dim, self.hidden_dim)
#         self.v = nn.Linear(v_dim, self.hidden_dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.proj_drop = nn.Dropout(dropout)
#
#     def forward(self, query, key, value):
#         q = self.q(query)
#         B, N, C = q.shape
#         q_res = q.clone()
#
#         k = self.k(key)
#         v = self.v(value)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         out = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         out = self.proj(out)
#         out = self.proj_drop(out)
#
#         return out * q_res
#
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 79.44 | 75.36 | 67.41 |  55.6 | 33.61 | 79.67 | 69.42 |
# +-------+-------+-------+-------+-------+-------+-------+



# class vg_neck(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.,
#                  mlp_ratio=4.,
#                  img_size=640,
#                  **kwargs):
#         super(vg_neck, self).__init__()
#
#         self.vl_att_layers = nn.ModuleList([
#             VLAttBlock(in_channel, l_dim, num_heads, dropout, mlp_ratio) for in_channel in in_channels
#         ])
#
#         self.lv_att_layers = nn.ModuleList([
#             LVAttBlock(l_dim, in_channel, num_heads, dropout, mlp_ratio) for in_channel in in_channels
#         ])
#
#         self.vllv_att_layers = nn.ModuleList([
#             VLLVAttBlock(in_channel, l_dim, num_heads, dropout, mlp_ratio) for in_channel in in_channels
#         ])
#
#     def forward(self, inputs, l, l_mask=None):
#         outs = []
#
#         for vl_layer, lv_layer, vllv_layer, x in zip(self.vl_att_layers, self.lv_att_layers, self.vllv_att_layers,
#                                                      inputs):
#             B, C, H, W = x.shape
#             x = x.flatten(2).transpose(1, 2)
#             mm = vllv_layer(vl_layer(x, l), lv_layer(l, x))
#             out = mm.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 78.86 | 74.56 | 66.16 |  54.8 | 32.86 | 77.88 | 68.17 |
# +-------+-------+-------+-------+-------+-------+-------+


# class vg_neck(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.,
#                  mlp_ratio=4.,
#                  img_size=640,
#                  **kwargs):
#         super(vg_neck, self).__init__()
#
#         self.lang_proj_layers = nn.ModuleList([
#             nn.Linear(l_dim, in_channel) for in_channel in in_channels
#         ])
#
#         self.vl_att_layers = nn.ModuleList([
#             VLAttBlock(in_channel, in_channel, num_heads, dropout, mlp_ratio) for in_channel in in_channels
#         ])
#
#         self.lv_att_layers = nn.ModuleList([
#             LVAttBlock(in_channel, in_channel, num_heads, dropout, mlp_ratio) for in_channel in in_channels
#         ])
#
#         self.vllv_att_layers = nn.ModuleList([
#             VLLVAttBlock(in_channel, in_channel, num_heads, dropout, mlp_ratio) for in_channel in in_channels
#         ])
#
#     def forward(self, inputs, l, l_mask=None):
#         outs = []
#
#         for l_proj_layer, vl_layer, lv_layer, vllv_layer, x in zip(self.lang_proj_layers, self.vl_att_layers,
#                                                                    self.lv_att_layers, self.vllv_att_layers, inputs):
#             B, C, H, W = x.shape
#             x = x.flatten(2).transpose(1, 2)
#             l_proj = l_proj_layer(l)
#             mm = vllv_layer(vl_layer(x, l_proj), lv_layer(l_proj, x))
#             out = mm.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
# +-------+-------+-------+-------+-------+-------+------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU | mIoU |
# +-------+-------+-------+-------+-------+-------+------+
# | 80.46 | 76.51 | 69.36 | 56.93 | 33.61 | 79.13 | 69.9 |
# +-------+-------+-------+-------+-------+-------+------+




# class vg_neck(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.,
#                  mlp_ratio=4.,
#                  img_size=640,
#                  **kwargs):
#         super(vg_neck, self).__init__()
#
#         self.lang_proj_layers = nn.ModuleList([
#             nn.Linear(l_dim, in_channel) for in_channel in in_channels
#         ])
#
#         self.vl_att_layers = nn.ModuleList([
#             VLAttBlock(in_channel, l_dim, num_heads, dropout, mlp_ratio) for in_channel in in_channels
#         ])
#
#         self.lv_att_layers = nn.ModuleList([
#             LVAttBlock(in_channel, in_channel, num_heads, dropout, mlp_ratio) for in_channel in in_channels
#         ])
#
#         self.vllv_att_layers = nn.ModuleList([
#             VLLVAttBlock(in_channel, in_channel, num_heads, dropout, mlp_ratio) for in_channel in in_channels
#         ])
#
#     def forward(self, inputs, l, l_mask=None):
#         outs = []
#
#         for l_proj_layer, vl_layer, lv_layer, vllv_layer, x in zip(self.lang_proj_layers, self.vl_att_layers,
#                                                                    self.lv_att_layers, self.vllv_att_layers, inputs):
#             B, C, H, W = x.shape
#             x = x.flatten(2).transpose(1, 2)
#             l_proj = l_proj_layer(l)
#             mm = vllv_layer(vl_layer(x, l), lv_layer(l_proj, x))
#             out = mm.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 79.22 | 75.27 | 68.29 | 55.46 | 33.39 | 79.31 | 68.63 |
# +-------+-------+-------+-------+-------+-------+-------+
#
#
# class VLAttBlock(nn.Module):
#     def __init__(self, dim, l_dim, num_heads=8, dropout=0., mlp_ratio=4.):
#         super(VLAttBlock, self).__init__()
#
#         self.norm_vis = nn.LayerNorm(dim)
#         self.norm_lang = nn.LayerNorm(l_dim)
#         self.cross_attention = CAM(dim, l_dim, num_heads, dropout)
#
#         self.norm_ffn = nn.LayerNorm(dim)
#         self.ffn = FFN(dim, int(dim * mlp_ratio))
#
#     def forward(self, x, l):
#         out = x + self.cross_attention(self.norm_vis(x), self.norm_lang(l))
#         out = out + self.ffn(self.norm_ffn(out))
#
#         return out
#
#
# class LVAttBlock(nn.Module):
#     def __init__(self, l_dim, dim, num_heads=8, dropout=0., mlp_ratio=4.):
#         super(LVAttBlock, self).__init__()
#
#         self.norm_lang = nn.LayerNorm(l_dim)
#         self.norm_vis = nn.LayerNorm(dim)
#         self.cross_attention = CAM(l_dim, dim, num_heads, dropout)
#
#         self.norm_ffn = nn.LayerNorm(l_dim)
#         self.ffn = FFN(l_dim, int(l_dim * mlp_ratio))
#
#     def forward(self, l, x):
#         out = l + self.cross_attention(self.norm_lang(l), self.norm_vis(x))
#         out = out + self.ffn(self.norm_ffn(out))
#
#         return out
#
#
# class VLLVAttBlock(nn.Module):
#     def __init__(self, dim, l_dim, num_heads=8, dropout=0., mlp_ratio=4.):
#         super(VLLVAttBlock, self).__init__()
#
#         self.norm_vl = nn.LayerNorm(dim)
#         self.norm_lv = nn.LayerNorm(l_dim)
#         self.cross_attention = CAM(dim, l_dim, num_heads, dropout)
#
#         self.norm_ffn = nn.LayerNorm(dim)
#         self.ffn = FFN(dim, int(dim * mlp_ratio))
#
#     def forward(self, xl, lx):
#         out = xl + self.cross_attention(self.norm_vl(xl), self.norm_lv(lx))
#         out = out + self.ffn(self.norm_ffn(out))
#
#         return out
#
#
# class CAM(nn.Module):
#     def __init__(self, q_dim, kv_dim, num_heads, dropout):
#         super(CAM, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = q_dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q = nn.Linear(q_dim, q_dim)
#         self.k = nn.Linear(kv_dim, q_dim)
#         self.v = nn.Linear(kv_dim, q_dim)
#
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(q_dim, q_dim)
#         self.proj_drop = nn.Dropout(dropout)
#
#     def forward(self, query, key_value):
#         B, N, C = query.shape
#
#         q = self.q(query)
#         k = self.k(key_value)
#         v = self.v(key_value)
#
#         q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         out = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         out = self.proj(out)
#         out = self.proj_drop(out)
#
#         out = out * query
#
#         return out
#
#
# class FFN(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(dropout)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#
#         return x




# class vg_neck(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.,
#                  **kwargs):
#         super(vg_neck, self).__init__()
#
#         self.vis_proj_layers = nn.ModuleList([
#             CBR(in_channel, in_channel, 3, 1, 1) for in_channel in in_channels
#         ])
#
#         self.lang_proj_layers = nn.ModuleList([
#             nn.Sequential(nn.Linear(l_dim, l_dim),
#                           nn.GELU(),
#                           nn.Dropout(dropout)) for _ in in_channels
#         ])
#
#         self.fusion_layers = nn.ModuleList([CAM(in_channel, l_dim, num_heads, dropout) for in_channel in in_channels])
#
#     def forward(self, inputs, l, l_mask):
#         outs = []
#
#         for vis_proj_layer, lang_proj_layer, fusion_layer, x in zip(self.vis_proj_layers, self.lang_proj_layers,
#                                                                     self.fusion_layers, inputs):
#             B, C, H, W = x.shape
#             mm = fusion_layer(vis_proj_layer(x).flatten(2).transpose(1, 2), lang_proj_layer(l), l_mask)
#             out = mm.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs
#
#
#
# class CAM(nn.Module):
#     def __init__(self, dim, l_dim, num_heads, dropout):
#         super(CAM, self).__init__()
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
#         x_res = x.clone()
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
#         out = mm * x_res
#
#         return out
#
# +-------+-------+-------+-------+-------+-------+-------+
# | PR@.5 | PR@.6 | PR@.7 | PR@.8 | PR@.9 |  oIoU |  mIoU |
# +-------+-------+-------+-------+-------+-------+-------+
# | 78.11 | 74.16 |  65.9 | 53.46 | 30.91 | 78.45 | 67.83 |
# +-------+-------+-------+-------+-------+-------+-------+



# v11
# class vg_neck(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.,
#                  in_indexes=[0, 1, 2, 3, 4],
#                  **kwargs):
#         super(vg_neck, self).__init__()
#
#         self.in_indexes = in_indexes
#
#         if in_indexes[-1] != 4:
#             self.vis_proj_layers = nn.ModuleList([
#                 CBR(in_channels[in_index], out_channels, 1) for in_index in in_indexes
#             ])
#         else:
#             self.vis_proj_layers = nn.ModuleList([
#                 CBR(in_channels[in_index], out_channels, 1) for in_index in in_indexes[:-1]
#             ])
#             self.vis_proj_layers.append(CBR(in_channels[-1], out_channels, 3, 2, 1))
#
#
#         self.lang_proj_layers = nn.ModuleList([
#             nn.Sequential(nn.Linear(l_dim, out_channels),
#                           nn.GELU(),
#                           nn.Dropout(dropout)) for _ in in_indexes
#         ])
#
#         self.fusion_layers = nn.ModuleList([CAM(out_channels, out_channels, num_heads, dropout) for _ in in_indexes])
#
#     def forward(self, inputs, l, l_mask=None):
#         outs = []
#
#         for vis_proj_layer, lang_proj_layer, fusion_layer, in_index in zip(self.vis_proj_layers, self.lang_proj_layers,
#                                                                            self.fusion_layers, self.in_indexes):
#             if in_index != 4:
#                 x = inputs[in_index]
#             else:
#                 x = inputs[-1]
#
#             x = vis_proj_layer(x)
#             B, C, H, W = x.shape
#             mm = fusion_layer(x.flatten(2).transpose(1, 2), lang_proj_layer(l), l_mask)
#             out = mm.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs

# v12
# class vg_neck(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.,
#                  in_indexes=[0, 1, 2, 3, 4],
#                  **kwargs):
#         super(vg_neck, self).__init__()
#
#         self.in_indexes = in_indexes
#
#         if in_indexes[-1] != 4:
#             self.vis_proj_layers = nn.ModuleList([
#                 nn.Sequential(
#                     nn.Conv2d(in_channels[in_index], out_channels, 1),
#                     nn.GroupNorm(32, out_channels)
#                 ) for in_index in in_indexes
#             ])
#         else:
#             self.vis_proj_layers = nn.ModuleList([
#                 nn.Sequential(
#                     nn.Conv2d(in_channels[in_index], out_channels, 1),
#                     nn.GroupNorm(32, out_channels)
#                 ) for in_index in in_indexes[:-1]
#             ])
#             self.vis_proj_layers.append(nn.Sequential(
#                     nn.Conv2d(in_channels[-1], out_channels, 3, 2, 1),
#                     nn.GroupNorm(32, out_channels)
#                 ))
#
#         self.lang_proj_layers = nn.ModuleList([
#             nn.Sequential(nn.Linear(l_dim, out_channels),
#                           nn.GELU(),
#                           nn.Dropout(dropout)) for _ in in_indexes
#         ])
#
#         self.fusion_layers = nn.ModuleList([CAM(out_channels, out_channels, num_heads, dropout) for _ in in_indexes])
#
#     def forward(self, inputs, l, l_mask=None):
#         outs = []
#
#         for vis_proj_layer, lang_proj_layer, fusion_layer, in_index in zip(self.vis_proj_layers, self.lang_proj_layers,
#                                                                            self.fusion_layers, self.in_indexes):
#             if in_index != 4:
#                 x = inputs[in_index]
#             else:
#                 x = inputs[-1]
#
#             x = vis_proj_layer(x)
#             B, C, H, W = x.shape
#             mm = fusion_layer(x.flatten(2).transpose(1, 2), lang_proj_layer(l), l_mask)
#             out = mm.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#             outs.append(out)
#
#         return outs

# v13
from rsfm.decoder.vg.lqvg.module import FeatureResizer

class vg_neck(nn.Module):
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 out_channels=256,
                 l_dim=768,
                 num_heads=8,
                 dropout=0.,
                 num_feature_levels=4,
                 **kwargs):
        super(vg_neck, self).__init__()
        self.num_feature_levels = num_feature_levels

        self.lang_proj = FeatureResizer(l_dim, out_channels, 0.1)

        if num_feature_levels > 1:
            num_backbone_outs = num_feature_levels - 1
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channel = in_channels[-num_backbone_outs:][_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channel, out_channels, kernel_size=1),
                    nn.GroupNorm(32, out_channels),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channel, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, out_channels),
                ))
                in_channel = out_channels
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels[-1], out_channels, kernel_size=1),
                    nn.GroupNorm(32, out_channels),
                )])

        self.fusion = CAM(out_channels, out_channels, num_heads, dropout)

    def forward(self, inputs, l, l_mask=None):
        outs, vis_inputs = [], []
        l = self.lang_proj(l)
        for x in inputs[(1 - self.num_feature_levels):]:
            vis_inputs.append(x)
        vis_inputs.append(inputs[-1])
        for ind, vis_input in enumerate(vis_inputs):
            x = self.input_proj[ind](vis_input)
            B, C, H, W = x.shape
            mm = self.fusion(x.flatten(2).transpose(1, 2), l, l_mask)
            out = mm.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(out)

        return outs

class CAM(nn.Module):
    def __init__(self, dim, l_dim, num_heads, dropout):
        super(CAM, self).__init__()
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

        out = mm * x_res

        return out

# v14
# from rsfm.decoder.vg.lqvg.module import FeatureResizer, VisionLanguageFusionModule, PositionEmbeddingSine1D, NestedTensor
#
# class vg_neck(nn.Module):
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  out_channels=256,
#                  l_dim=768,
#                  num_heads=8,
#                  dropout=0.,
#                  num_feature_levels=4,
#                  **kwargs):
#         super(vg_neck, self).__init__()
#         self.num_feature_levels = num_feature_levels
#
#         self.lang_proj = FeatureResizer(l_dim, out_channels, 0.1)
#         self.lang_pos = PositionEmbeddingSine1D(out_channels, normalize=True)
#
#         if num_feature_levels > 1:
#             num_backbone_outs = num_feature_levels - 1
#             input_proj_list = []
#             for _ in range(num_backbone_outs):
#                 in_channel = in_channels[-num_backbone_outs:][_]
#                 input_proj_list.append(nn.Sequential(
#                     nn.Conv2d(in_channel, out_channels, kernel_size=1),
#                     nn.GroupNorm(32, out_channels),
#                 ))
#             for _ in range(num_feature_levels - num_backbone_outs):
#                 input_proj_list.append(nn.Sequential(
#                     nn.Conv2d(in_channel, out_channels, kernel_size=3, stride=2, padding=1),
#                     nn.GroupNorm(32, out_channels),
#                 ))
#                 in_channel = out_channels
#             self.input_proj = nn.ModuleList(input_proj_list)
#         else:
#             self.input_proj = nn.ModuleList([
#                 nn.Sequential(
#                     nn.Conv2d(in_channels[-1], out_channels, kernel_size=1),
#                     nn.GroupNorm(32, out_channels),
#                 )])
#
#         self.fusion = VisionLanguageFusionModule(out_channels, num_heads, dropout)
#
#     def forward(self, inputs, l, l_mask):
#         outs, vis_inputs = [], []
#
#         l = self.lang_proj(l)
#         l = NestedTensor(l, l_mask)
#         l_pos = self.lang_pos(l).permute(2, 0, 1)
#         l, l_mask = l.decompose()
#         l = l.permute(1, 0, 2)
#
#         for x in inputs[(1 - self.num_feature_levels):]:
#             vis_inputs.append(x)
#         vis_inputs.append(inputs[-1])
#         for ind, vis_input in enumerate(vis_inputs):
#             x = self.input_proj[ind](vis_input)
#             B, C, H, W = x.shape
#             mm = self.fusion(tgt=x.flatten(2).permute(2, 0, 1),
#                              memory=l,
#                              memory_key_padding_mask=l_mask,
#                              pos=l_pos,
#                              query_pos=None)
#             out = mm.permute(1, 2, 0).reshape(B, -1, H, W).contiguous()
#             outs.append(out)
#
#         return outs