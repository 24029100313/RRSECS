import torch
from torch import nn
from torch.nn import functional as F


class LAVTHead(nn.Module):
    '''
    from 'LAVT: Language-Aware Vision Transformer for Referring Image Segmentation'
    '''
    def __init__(self,
                 in_channels=[128, 256, 512, 1024],
                 embedding_dim=512,
                 num_classes=2,
                 **kwargs):
        super(LAVTHead, self).__init__()

        embedding_dim = in_channels[-2]
        c1_size, c2_size, c3_size, c4_size = in_channels

        self.conv1_4 = nn.Conv2d(c4_size+c3_size, embedding_dim, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(embedding_dim)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(embedding_dim)
        self.relu2_4 = nn.ReLU()

        self.conv1_3 = nn.Conv2d(embedding_dim + c2_size, embedding_dim, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(embedding_dim)
        self.relu1_3 = nn.ReLU()
        self.conv2_3 = nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(embedding_dim)
        self.relu2_3 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(embedding_dim + c1_size, embedding_dim, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(embedding_dim)
        self.relu1_2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(embedding_dim)
        self.relu2_2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, inputs):
        x_c1, x_c2, x_c3, x_c4 = inputs

        # fuse Y4 and Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x)
        # fuse top-down features and Y2 features
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x)
        # fuse top-down features and Y1 features
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        return self.conv1_1(x)
