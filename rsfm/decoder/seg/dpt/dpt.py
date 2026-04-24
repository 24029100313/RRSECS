import torch.nn as nn
import torch.nn.functional as F
from .utils import _make_fusion_block, _make_scratch



class DPTHead(nn.Module):
    '''
    Vision Transformers for Dense Prediction, ICCV 2021
    '''
    def __init__(self,
                 num_classes=7,
                 in_channels=[96, 192, 384, 768],
                 embedding_dim=96,
                 use_bn=False,
                 use_depth=False,
                 **kwargs):
        super(DPTHead, self).__init__()

        self.use_depth = use_depth

        self.scratch = _make_scratch(
            in_channels,
            embedding_dim,
            groups=1,
            expand=False,
        )

        self.scratch.refinenet1 = _make_fusion_block(embedding_dim, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(embedding_dim, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(embedding_dim, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(embedding_dim, use_bn)

        if not use_depth:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(embedding_dim, num_classes, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(embedding_dim, embedding_dim // 2, kernel_size=3, stride=1, padding=1)

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(embedding_dim // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )


    def forward(self, inputs):
        layer_1, layer_2, layer_3, layer_4 = inputs

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if not self.use_depth:
            out = self.scratch.output_conv(path_1)
        else:
            out = self.scratch.output_conv1(path_1)
            out = F.interpolate(out, (int(layer_3.shape[2] * 14), int(layer_3.shape[3] * 14)),
                                mode="bilinear", align_corners=True)
            out = self.scratch.output_conv2(out)

        return out