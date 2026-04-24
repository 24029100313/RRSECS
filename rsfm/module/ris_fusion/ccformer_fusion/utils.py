import math
import torch
from torch import nn



class FusionLayer(nn.Module):
    def __init__(self,
                 v_dim=96,
                 l_dim=768,
                 dim=96,
                 num_heads=4,
                 dropout=0.):
        super(FusionLayer, self).__init__()

        self.l_dim = l_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        self.v_q = nn.Sequential(nn.Conv1d(v_dim, self.dim, kernel_size=1, stride=1),
                                 nn.InstanceNorm1d(self.dim),
                                 nn.Dropout(dropout))

        self.l_q = nn.Linear(self.l_dim, self.dim)

        self.values_l = nn.Linear(self.l_dim, self.dim)

        self.w_v = nn.Sequential(nn.Conv1d(self.dim, self.v_dim, kernel_size=1, stride=1),
                                 nn.InstanceNorm1d(self.v_dim))

        self.project_mm = nn.Sequential(nn.Conv1d(v_dim, v_dim, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, l, l_mask):
        # x shape: (B, H*W, v_dim)
        # l input shape: (B, N_l, l_dim)
        # l_mask shape: (B, 1, N_l)

        B, HW = v.size(0), v.size(1)

        v = v.permute(0, 2, 1)
        query_v = self.v_q(v)  # (B, dim, H*W)
        key_l = self.l_q(l)  # (B, N_l, dim)

        v_query = query_v.reshape(B, self.num_heads, self.head_dim, HW).permute(0, 1, 3, 2)  # (B, num_head, HW, dim//num_heads)
        l_key = self.transpose_for_scores(key_l)  # (B, num_head, N_l, dim//num_heads)

        value_l = self.values_l(l)  # (B, N_l, dim)

        l_value = self.transpose_for_scores(value_l)  # (B, num_head, N_l, dim//num_heads)

        # Query : Vision , Key : Language
        attention_scores_v = torch.matmul(v_query, l_key.transpose(-1, -2))  # (B, num_head, HW, N_l)
        attention_scores_v = attention_scores_v / math.sqrt(self.head_dim)

        l_mask = l_mask.unsqueeze(1)  # (B, 1, 1, N_l)
        l_mask = 1e4 * l_mask - 1e4

        attn_v = attention_scores_v + l_mask
        attn_v = nn.Softmax(dim=-1)(attn_v)

        out_v = torch.matmul(attn_v, l_value)  # (B, num_head, HW, dim//num_head)
        out_v = out_v.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.dim)

        w_v = self.w_v(out_v.permute(0, 2, 1))
        attn_output_v = w_v + v
        attn_output_v = self.project_mm(attn_output_v)
        attn_output_v = attn_output_v.permute(0, 2, 1)

        return attn_output_v