import math
import torch
import torch.nn as nn
from rsfm.module.neck.mscab.utils import CrossAttentionModule
from rsfm.utils import inverse_sigmoid, get_clones, get_activation_fn
from rsfm.decoder.od.deformable_detr.ops.modules import MSDeformAttn



class CrossAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, dim, num_heads, dropout):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(q_dim, dim)
        self.k = nn.Linear(k_dim, dim)
        self.v = nn.Linear(v_dim, dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_mm = nn.Linear(dim, dim)
        self.proj_mm_drop = nn.Dropout(dropout)

    def forward(self, query, key, value):
        if len(query.shape) == 4:
            vis_query = True
            query = query.flatten(2).permute(0, 2, 1)
        elif len(query.shape) == 3:
            vis_query = False
            assert len(key.shape) == 4
            key = key.flatten(2).permute(0, 2, 1)
            value = value.flatten(2).permute(0, 2, 1)
        else:
            raise NotImplementedError

        B, N, C = query.shape
        query_res = query.clone()

        q = self.q(query)
        k = self.k(key)
        v = self.v(value)

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        mm = (attn @ v).transpose(1, 2).reshape(B, N, C)
        mm = self.proj_mm(mm)
        mm = self.proj_mm_drop(mm)

        out = query_res * mm

        if vis_query:
            H = W = int(N ** 0.5)
            out = out.permute(0, 2, 1).reshape(B, C, H, W)

        return out


class BVLIM(nn.Module):
    def __init__(self,
                 vis_dim=96,
                 lang_dim=768,
                 dim=256,
                 num_heads=8,
                 dropout=0.1):
        super(BVLIM, self).__init__()

        self.vis_proj = nn.Sequential(
            nn.Conv2d(vis_dim, dim, 1),
            nn.GroupNorm(32, dim))

        self.lang_proj = nn.Sequential(
            nn.Linear(lang_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout))

        self.vl_ca = CrossAttention(dim, dim, dim, dim, num_heads, dropout)

        self.lv_ca = CrossAttention(dim, dim, dim, dim, num_heads, dropout)

    def forward(self, vis, lang):
        '''
        vis: B, C, H, W
        Lang: B, N, C
        '''
        vis = self.vis_proj(vis).flatten(2).permute(0, 2, 1)
        lang = self.lang_proj(lang)

        vis_mm = self.vl_ca(vis, lang, lang)
        lang_mm = self.lv_ca(lang, vis, vis)

        return vis_mm, lang_mm


'''v3.0'''
# class MBVIM(nn.Module):
#     '''
#     MBVIM: multi-scale bidirectional vision-language interaction module
#     '''
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  lang_dim=768,
#                  embedding_dim=256,
#                  num_heads=8,
#                  dropout=0.1):
#         super(MBVIM, self).__init__()
#
#         self.vis_proj = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(in_channel, embedding_dim, 1),
#                 nn.GroupNorm(32, embedding_dim)
#             ) for in_channel in in_channels
#         ])
#
#         self.lang_proj = nn.Sequential(
#             nn.Linear(lang_dim, embedding_dim),
#             nn.LayerNorm(embedding_dim),
#             nn.Dropout(dropout)
#         )
#
#         self.vl_ca = nn.ModuleList([
#             CrossAttention(embedding_dim, embedding_dim, embedding_dim,
#                            embedding_dim, num_heads, dropout) for _ in range(4)
#         ])
#
#         self.lv_ca = nn.ModuleList([
#             CrossAttention(embedding_dim, embedding_dim, embedding_dim,
#                            embedding_dim, num_heads, dropout) for _ in range(4)
#         ])
#
#         self.task_gate = nn.ModuleList([Gate(embedding_dim) for _ in range(4)])
#
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#
#     def forward(self, features, l):
#         l = self.lang_proj(l.mean(dim=-1)).unsqueeze(1)
#         srcs = []
#         for i, feature in enumerate(features):
#             srcs.append(self.vis_proj[i](feature))
#
#         f1, f2, f3, f4 = srcs
#         f4_mm = self.vl_ca[3](f4, l, l)
#         l4_mm = self.lv_ca[3](l, f4, f4)
#         f4_mm_ris, f4_mm_vg = self.task_gate[3](f4_mm)
#
#         f3 = self.up(f4_mm) + f3
#         f3_mm = self.vl_ca[2](f3, l4_mm, l4_mm)
#         l3_mm = self.lv_ca[2](l4_mm, f3, f3)
#         f3_mm_ris, f3_mm_vg = self.task_gate[2](f3_mm)
#
#         f2 = self.up(f3_mm) + f2
#         f2_mm = self.vl_ca[1](f2, l3_mm, l3_mm)
#         l2_mm = self.lv_ca[1](l3_mm, f2, f2)
#         f2_mm_ris, f2_mm_vg = self.task_gate[1](f2_mm)
#
#         f1 = self.up(f2_mm) + f1
#         f1_mm = self.vl_ca[0](f1, l2_mm, l2_mm)
#         l1_mm = self.lv_ca[0](l2_mm, f1, f1)
#         f1_mm_ris, f1_mm_vg = self.task_gate[0](f1_mm)
#
#         return [f1_mm_ris, f2_mm_ris, f3_mm_ris, f4_mm_ris], \
#                [f1_mm_vg, f2_mm_vg, f3_mm_vg, f4_mm_vg]


'''v3.1'''
class MBVIM(nn.Module):
    '''
    MBVIM: multi-scale bidirectional vision-language interaction module
    '''
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 lang_dim=768,
                 embedding_dim=256,
                 num_heads=8,
                 dropout=0.1):
        super(MBVIM, self).__init__()

        self.vis_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel, embedding_dim, 1),
                nn.GroupNorm(32, embedding_dim)
            ) for in_channel in in_channels
        ])

        self.lang_proj = nn.Sequential(
            nn.Linear(lang_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout)
        )

        self.vl_ca = nn.ModuleList([
            CrossAttention(embedding_dim, embedding_dim, embedding_dim,
                           embedding_dim, num_heads, dropout) for _ in range(4)
        ])

        self.lv_ca = nn.ModuleList([
            CrossAttention(embedding_dim, embedding_dim, embedding_dim,
                           embedding_dim, num_heads, dropout) for _ in range(4)
        ])

        self.fusion_conv = nn.ModuleList([
            nn.Conv2d(2 * embedding_dim, embedding_dim, 1) for _ in range(3)
        ])

        self.task_gate = nn.ModuleList([Gate(embedding_dim) for _ in range(4)])

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, features, l):
        l = self.lang_proj(l.mean(dim=-1)).unsqueeze(1)
        srcs = []
        for i, feature in enumerate(features):
            srcs.append(self.vis_proj[i](feature))

        f1, f2, f3, f4 = srcs
        f4_mm = self.vl_ca[3](f4, l, l)
        l4_mm = self.lv_ca[3](l, f4, f4)
        f4_mm_ris, f4_mm_vg = self.task_gate[3](f4_mm)

        f3 = self.fusion_conv[2](torch.cat((self.up(f4_mm), f3), dim=1))
        f3_mm = self.vl_ca[2](f3, l4_mm, l4_mm)
        l3_mm = self.lv_ca[2](l4_mm, f3, f3)
        f3_mm_ris, f3_mm_vg = self.task_gate[2](f3_mm)

        f2 = self.fusion_conv[1](torch.cat((self.up(f3_mm), f2), dim=1))
        f2_mm = self.vl_ca[1](f2, l3_mm, l3_mm)
        l2_mm = self.lv_ca[1](l3_mm, f2, f2)
        f2_mm_ris, f2_mm_vg = self.task_gate[1](f2_mm)

        f1 = self.fusion_conv[0](torch.cat((self.up(f2_mm), f1), dim=1))
        f1_mm = self.vl_ca[0](f1, l2_mm, l2_mm)
        l1_mm = self.lv_ca[0](l2_mm, f1, f1)
        f1_mm_ris, f1_mm_vg = self.task_gate[0](f1_mm)

        return [f1_mm_ris, f2_mm_ris, f3_mm_ris, f4_mm_ris], \
               [f1_mm_vg, f2_mm_vg, f3_mm_vg, f4_mm_vg]


# To do: 去除CCFormer融合，MBVIM先添加Top-Down文本融合, Bottom-Up词义融合


# class MBVIM(nn.Module):
#     '''
#     MBVIM: multi-scale bidirectional vision-language interaction module
#     '''
#     def __init__(self,
#                  in_channels=[96, 192, 384, 768],
#                  lang_dim=768,
#                  embedding_dim=256,
#                  num_heads=8,
#                  dropout=0.1):
#         super(MBVIM, self).__init__()
#
#         self.vis_proj = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(in_channel, embedding_dim, 1),
#                 nn.GroupNorm(32, embedding_dim)
#             ) for in_channel in in_channels
#         ])
#
#         self.lang_proj = nn.Sequential(
#             nn.Linear(lang_dim, embedding_dim),
#             nn.LayerNorm(embedding_dim),
#             nn.Dropout(dropout)
#         )
#
#         self.vl_ca = nn.ModuleList([
#             CrossAttention(embedding_dim, embedding_dim, embedding_dim,
#                            embedding_dim, num_heads, dropout) for _ in range(4)
#         ])
#
#         self.lv_ca = nn.ModuleList([
#             CrossAttention(embedding_dim, embedding_dim, embedding_dim,
#                            embedding_dim, num_heads, dropout) for _ in range(4)
#         ])
#
#         self.fusion_conv = nn.ModuleList([
#             nn.Conv2d(2 * embedding_dim, embedding_dim, 1) for _ in range(3)
#         ])
#
#         self.down_conv = nn.ModuleList([
#             nn.Conv2d(embedding_dim, embedding_dim, 3, 2, 1) for _ in range(3)
#         ])
#
#         self.task_gate = nn.ModuleList([Gate(embedding_dim) for _ in range(4)])
#
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#
#     def forward(self, features, l):
#         l = self.lang_proj(l.permute(0, 2, 1))
#         srcs = []
#         for i, feature in enumerate(features):
#             srcs.append(self.vis_proj[i](feature))
#
#         f1, f2, f3, f4 = srcs
#         f4_mm = self.vl_ca[3](f4, l, l)
#         l4_mm = self.lv_ca[3](l, f4, f4)
#
#         f3 = self.fusion_conv[2](torch.cat((self.up(f4_mm), f3), dim=1))
#         f3_mm = self.vl_ca[2](f3, l4_mm, l4_mm)
#         l3_mm = self.lv_ca[2](l4_mm, f3, f3)
#
#         f2 = self.fusion_conv[1](torch.cat((self.up(f3_mm), f2), dim=1))
#         f2_mm = self.vl_ca[1](f2, l3_mm, l3_mm)
#         l2_mm = self.lv_ca[1](l3_mm, f2, f2)
#
#         f1 = self.fusion_conv[0](torch.cat((self.up(f2_mm), f1), dim=1))
#         f1_mm = self.vl_ca[0](f1, l2_mm, l2_mm)
#         l1_mm = self.lv_ca[0](l2_mm, f1, f1)
#
#         l_word = torch.mean(l1_mm, dim=1)
#
#
#
#
#         return [f1_mm_ris, f2_mm_ris, f3_mm_ris, f4_mm_ris], \
#                [f1_mm_vg, f2_mm_vg, f3_mm_vg, f4_mm_vg]


class Gate(nn.Module):
    def __init__(self, dim):
        super(Gate, self).__init__()

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1) # v3, v3.1, v4

        # self.conv = nn.Sequential(
        #     nn.Conv2d(dim, dim, 3, 1, 1),
        #     nn.GroupNorm(32, dim)) # v3.2

        self.weight = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        w = self.weight(x)
        x_ris = w * x
        x_vg = (1 - w) * x

        return x_ris, x_vg





class LAGD(nn.Module):
    def __init__(self, v_dim, l_dim, n_heads=8, dropout=0.):
        super(LAGD, self).__init__()

        self.cmf = CrossAttentionModule(v_dim, l_dim, n_heads, dropout)

        self.gate = nn.Sequential(
            nn.Conv2d(v_dim, v_dim, 1),
            nn.BatchNorm2d(v_dim),
            nn.Sigmoid())

    def forward(self, x, l, l_mask):
        B, _, H, W = x.shape
        x = self.cmf(x.flatten(2).transpose(1, 2), l.permute(0, 2, 1))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        g = self.gate(x)
        rrsis_feat = g * x
        rsvg_feat = (1 - g) * x

        return rrsis_feat, rsvg_feat


class MSDeformAttnTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class MSDeformAttnTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class MSDeformAttnTransformerVG(nn.Module):
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 return_intermediate_dec=False,
                 num_feature_levels=4,
                 dec_n_points=4,
                 two_stage_num_proposals=1):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage_num_proposals = two_stage_num_proposals

        decoder_layer = MSDeformAttnTransformerDecoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, dec_n_points)
        self.decoder = MSDeformAttnTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        # two stage
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
        self.pos_trans_norm = nn.LayerNorm(d_model * 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]

        src_flatten = []
        mask_flatten = []
        spatial_shapes = []

        for lvl, (src, mask) in enumerate(zip(srcs, masks)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        bs, _, c = src_flatten.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(src_flatten, mask_flatten, spatial_shapes)

        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact.sigmoid()
        init_reference_out = reference_points
        pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
        query_embed, tgt = torch.split(pos_trans_out, c, dim=2)

        hs, inter_references = self.decoder(tgt, reference_points, src_flatten,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references

        return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact