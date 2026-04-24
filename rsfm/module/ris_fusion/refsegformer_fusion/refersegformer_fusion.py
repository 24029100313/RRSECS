import torch
from torch import nn
from .utils import ScaledDotProductAttention



class RefSegformer_fusion(nn.Module):
    '''
    from 'Towards Robust Referring Image Segmentation'
    '''
    def __init__(self, dim, l_dim, num_heads=8, dropout=0.0, num_mem=20, num_neg_mem=10):
        super().__init__()

        self.memory_token = nn.Embedding(num_mem, dim)
        self.neg_memory_token = nn.Embedding(num_neg_mem, dim)

        self.input_proj = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim),
        )
        self.lan_proj = nn.Sequential(
            nn.Conv1d(l_dim, dim, kernel_size=1, stride=1),
        )

        self.vision_lan_fuse = ScaledDotProductAttention(dim, h=num_heads, dropout=dropout)
        self.memory_fuse = ScaledDotProductAttention(dim, h=num_heads, dropout=dropout)
        self.feature_fuse = ScaledDotProductAttention(dim, h=num_heads, dropout=dropout)

        self.norm_vlf = nn.InstanceNorm1d(dim)
        self.norm_mf = nn.InstanceNorm1d(dim)
        self.norm_ff = nn.InstanceNorm1d(dim)

        self.output_proj = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(dim)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.output_proj[1].weight)

    def forward(self, x, l, l_mask):
        B, HW, C = x.shape
        x = self.input_proj(x.permute(0, 2, 1))  # B, C, HW
        x = x.permute(0, 2, 1)  # B, HW, C

        l = self.lan_proj(l)  # B, C, Nl
        l = l.permute(0, 2, 1)  # B, Nl, C

        vision_lan_fuse, _ = self.vision_lan_fuse(l, x, x, query_mask=l_mask)
        vision_lan_fuse = self.norm_vlf(vision_lan_fuse.permute(0, 2, 1)).permute(0, 2, 1)

        memory_token = self.memory_token.weight.unsqueeze(0).repeat(B, 1, 1)  # B, Nm, C
        lan_mem, _ = self.memory_fuse(memory_token, vision_lan_fuse, vision_lan_fuse, key_mask=l_mask)

        neg_memory_token = self.neg_memory_token.weight.unsqueeze(0).repeat(B, 1, 1)  # B, Nnm, C
        lan_mem = torch.cat([neg_memory_token, lan_mem], dim=1)

        if lan_mem.shape[1] > 1:
            lan_mem = self.norm_mf(lan_mem.permute(0, 2, 1)).permute(0, 2, 1)

        x, _ = self.feature_fuse(x, lan_mem, lan_mem)  # B, HW, C
        x = self.norm_ff(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.output_proj(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x, x


