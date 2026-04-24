from torch import nn
from .utils import GatedCrossModalAttention



class SLViT_fusion(nn.Module):
    '''
    SLViT: Scale-Wise Language-Guided Vision Transformer for Referring Image Segmentation
    '''
    def __init__(self,
                 dim=96,
                 l_dim=768,
                 num_heads=1,
                 dropout=0.):
        super(SLViT_fusion, self).__init__()

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

        self.fusion = GatedCrossModalAttention(dim, dim, l_dim, dim, dim,
                                               num_heads=num_heads,
                                               dropout=dropout)

    def forward(self, x, l, l_mask):
        # visual input shape: (B, C_v, H, W)
        # linguistic input shape: (B, C_l, T)

        u = x.clone()
        attn = self.conv0(x)
        B, _, H, W = attn.shape

        # Convolutional Branches
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        # Cross-Modal Branch
        # visual input shape is changed to (B, H*W, C_v)
        attn_3, sim_temp = self.fusion(attn.reshape(B, -1, H* W).permute(0, 2, 1).contiguous(), l, l_mask)
        attn_3 = attn_3.permute(0, 2, 1).contiguous().reshape(B, -1, H, W)  # shape reverse (B, C_v, H, W)

        # Integrated Attention
        attn = self.conv3(attn + attn_0 + attn_1 + attn_2 + attn_3)

        out = attn * u

        return out, out
