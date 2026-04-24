import torch
from torch import nn
from torch.nn import functional as F
from .utils import MSB



class LGCEHead(nn.Module):
    '''
    from 'RRSIS: Referring Remote Sensing Image Segmentation'
    '''
    def __init__(self,
                 in_channels=[128, 256, 512, 1024],
                 embedding_dim=512,
                 num_classes=2, **kwargs):
        super(LGCEHead, self).__init__()

        embedding_dim = in_channels[-2]
        c1_size, c2_size, c3_size, c4_size = in_channels

        self.conv1_4 = nn.Conv2d(c4_size + c3_size, embedding_dim, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(embedding_dim)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(embedding_dim)
        self.relu2_4 = nn.ReLU()

        self.msb = MSB((in_channels[-2], in_channels[-1]))

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

        L = 768
        self.lg_layer3 = nn.Linear(L, in_channels[-2])
        self.lg_layer4 = nn.Linear(L, in_channels[-1])

    def forward(self, inputs, lguide):
        x_c1, x_c2, x_c3, x_c4 = inputs

        # fuse Y4 and Y3
        B, C4, H4, W4 = x_c4.shape
        B, C3, H3, W3 = x_c3.shape
        lguide = lguide.mean(-1)

        lg3 = self.lg_layer3(lguide).view([B, 1, C3])
        lg4 = self.lg_layer4(lguide).view([B, 1, C4])

        guide_tokens1 = [lg3, lg4]

        out1 = self.msb([x_c3.view(B, C3, -1).transpose(1, 2), x_c4.view(B, C4, -1).transpose(1, 2)], guide_tokens1)
        x_c3 = out1[0][:, 1:, :]
        x_c4 = out1[1][:, 1:, :]
        x_c3 = x_c3.transpose(1, 2).view([B, C3, H3, W3])
        x_c4 = x_c4.transpose(1, 2).view([B, C4, H4, W4])

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
