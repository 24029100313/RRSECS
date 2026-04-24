import torch
import torch.nn as nn
from functools import partial
from mmcv.runner import load_state_dict
from timm.models.layers import trunc_normal_
from rsfm.module import build_ris_fusion, CIM, build_vg_fusion
from .utils import LayerNorm, UniRepLKNetBlock



class UniRepLKNet(nn.Module):
    """
    UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio Video Point Cloud Time-Series and Image Recognition, CVPR 2024
    """
    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 depths=(3, 3, 27, 3),
                 embed_dim=96,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 kernel_sizes=None,
                 deploy=False,
                 with_cp=False,
                 attempt_use_lk_impl=True,
                 use_sync_bn=False,
                 l_dim=768,
                 vlf_ris=None,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.,
                 vlf_vg=None):
        super().__init__()

        self.in_channels = in_channels
        embed_dims = [embed_dim * 2 ** i for i in range(len(depths))]

        depths = tuple(depths)
        if kernel_sizes is None:
            if embed_dim == 80:
                kernel_sizes = ((3, 3, 3),
                                (13, 13, 13),
                                (13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3),
                                (13, 13, 13))
            elif embed_dim in [96, 128, 192, 256]:
                kernel_sizes = ((3, 3, 3),
                                (13, 13, 13),
                                (13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3),
                                (13, 13, 13))
            else:
                raise ValueError('no default kernel size settings for the given depths, '
                                 'please specify kernel sizes for each block, e.g., '
                                 '((3, 3), (13, 13), (13, 13, 13, 13, 13, 13), (13, 13))')

        for i in range(4):
            assert len(kernel_sizes[i]) == depths[i], 'kernel sizes do not match the depths'

        self.with_cp = with_cp

        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0] // 2, kernel_size=3, stride=2, padding=1),
            LayerNorm(embed_dims[0] // 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(embed_dims[0] // 2, embed_dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm(embed_dims[0], eps=1e-6, data_format="channels_first")))

        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                nn.Conv2d(embed_dims[i], embed_dims[i + 1], kernel_size=3, stride=2, padding=1),
                LayerNorm(embed_dims[i + 1], eps=1e-6, data_format="channels_first")))

        self.stages = nn.ModuleList()

        cur = 0
        for i in range(4):
            main_stage = nn.Sequential(
                *[UniRepLKNetBlock(dim=embed_dims[i], kernel_size=kernel_sizes[i][j], drop_path=dp_rates[cur + j],
                                   layer_scale_init_value=layer_scale_init_value, deploy=deploy,
                                   attempt_use_lk_impl=attempt_use_lk_impl,
                                   with_cp=with_cp, use_sync_bn=use_sync_bn) for j in
                  range(depths[i])])
            self.stages.append(main_stage)
            cur += depths[i]

            if self.vlf_ris:
                fusion = build_ris_fusion(vlf_ris, embed_dims[i], l_dim, num_heads_fusion[i], fusion_drop,
                                          size=img_size // (2 ** (i + 2)))
                setattr(self, f"fusion{i}", fusion)

            if self.vlf_vg:
                fusion = build_vg_fusion(vlf_vg, embed_dims[i], l_dim, size=img_size // (2 ** (i + 2)))
                setattr(self, f"fusion{i}", fusion)

        if self.vlf_ris == 'RMSIN':
            self.cim = CIM(dim=sum(embed_dims), channels=embed_dims, height=img_size // 32, width=img_size // 32)

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(embed_dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)


    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            checkpoint = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in checkpoint:
                _state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                _state_dict = checkpoint['model']
            else:
                _state_dict = checkpoint
            load_state_dict(self, _state_dict, strict=False)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self, x, l=None, l_mask=None):
        B = x.shape[0]
        outs = []
        for stage_idx in range(4):
            x = self.downsample_layers[stage_idx](x)
            x = self.stages[stage_idx](x)

            if self.vlf_ris or self.vlf_vg:
                H, W = x.shape[2:]
                x = x.flatten(2).transpose(1, 2)
                fusion = getattr(self, f"fusion{stage_idx}")

                if self.vlf_ris == 'DMMI':
                    x, x_residual, l = fusion(x, l, l_mask)
                else:
                    x, x_residual = fusion(x, l, l_mask)

                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                x_residual = x_residual.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

                outs.append(self.__getattr__(f'norm{stage_idx}')(x_residual))
            else:
                outs.append(self.__getattr__(f'norm{stage_idx}')(x))

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        if self.vlf_ris == 'DMMI':
            return l, tuple(outs)
        else:
            return tuple(outs)


    def reparameterize_unireplknet(self):
        for m in self.modules():
            if hasattr(m, 'reparameterize'):
                m.reparameterize()
