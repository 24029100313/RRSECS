import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from rsfm.utils import build_position_encoding, NestedTensor
from .utils import MSDeformAttnTransformerEncoderOnly



class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(
        self,
        input_shape,
        transformer_dropout=0.1,
        transformer_nheads=8,
        transformer_dim_feedforward=2048,
        transformer_enc_layers=6,
        conv_dim=256,
        mask_dim=256,
        # deformable transformer encoder args
        transformer_in_features= ["res3", "res4", "res5"],
        common_stride=4,
    ):
        super().__init__()
        transformer_input_shape = {k: v for k, v in input_shape.items() if k in transformer_in_features}
        
        # this is the input shape of pixel decoder        
        self.in_features = [k for k, v in input_shape.items()]  # starting from "res3" to "res5"        
        self.feature_channels = [v.channel for k, v in input_shape.items()] # eg. [16, 64, 128, 256]
        
        # this is the input shape of transformer encoder (could use less features than pixel decoder        
        self.transformer_in_features = [k for k, v in transformer_input_shape.items()]  # starting from "res3" to "res5"
        transformer_in_channels = [v.channel for k, v in transformer_input_shape.items()] # eg. [64, 128, 256]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape.items()]  # to decide extra FPN layers

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res3)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )

        self.pe_layer = build_position_encoding(conv_dim, 'sine')

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = nn.Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)
        
        self.maskformer_num_feature_levels = 3  # always use 3 scales
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]): # res2 -> fpn
            lateral_conv = nn.Sequential(nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                                         nn.GroupNorm(32, conv_dim),
                                         nn.ReLU(inplace=True))

            output_conv = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, kernel_size=3,  stride=1,  padding=1),
                                        nn.GroupNorm(32, conv_dim),
                                        nn.ReLU(inplace=True))
            
            weight_init.c2_xavier_fill(lateral_conv[0])
            weight_init.c2_xavier_fill(output_conv[0])
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def forward(self, features):
        srcs = []
        pos = []
        # Reverse feature maps into top-down order (from low to high resolution), 'res5' -> 'res3'
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(
                NestedTensor(x, torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool))
            ))

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            out.append(y)

        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        # return self.mask_features(out[-1]), out[0], multi_scale_features
        return self.mask_features(out[-1]), out, multi_scale_features
