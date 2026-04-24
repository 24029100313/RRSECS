import torch
from torch import nn
import torch.nn.functional as F
from .utils import Transformer_Fusion



class DMMIHead(nn.Module):
    '''
    from 'Beyond One-to-One: Rethinking the Referring Image Segmentation, ICCV 2023'
    '''
    def __init__(self,
                 in_channels=[128, 256, 512, 1024],
                 embedding_dim=768,
                 num_classes=2,
                 **kwargs):
        super(DMMIHead, self).__init__()
        c1_size, c2_size, c3_size, c4_size = in_channels

        self.lang_proj = nn.Conv1d(768, embedding_dim, 1)

        self.conv1_4 = nn.Conv2d(c4_size + c3_size, embedding_dim, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(embedding_dim)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(embedding_dim)
        self.relu2_4 = nn.ReLU()

        self.transformer_fusion1 = Transformer_Fusion(dim=embedding_dim, nhead=8, num_layers=1)

        self.conv1_3 = nn.Conv2d(embedding_dim + c2_size, embedding_dim, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(embedding_dim)
        self.relu1_3 = nn.ReLU()
        self.conv2_3 = nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(embedding_dim)
        self.relu2_3 = nn.ReLU()

        self.transformer_fusion2 = Transformer_Fusion(dim=embedding_dim, nhead=8, num_layers=1)

        self.conv1_2 = nn.Conv2d(embedding_dim + c1_size, embedding_dim, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(embedding_dim)
        self.relu1_2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(embedding_dim)
        self.relu2_2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, inputs, lan_full):
        x_c1, x_c2, x_c3, x_c4 = inputs
        lan_full = self.lang_proj(lan_full)

        # fuse Y4 and Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)

        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x) # [B, 512, 30, 30]

        x = self.transformer_fusion1(x, lan_full)

        # fuse top-down features and Y2 features and pre1
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x) # [B, 512, 60, 60]

        x = self.transformer_fusion2(x, lan_full)

        # fuse top-down features and Y1 features
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x) # [B, 512, 120, 120]

        out = self.conv1_1(x)

        return out