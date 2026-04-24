from functools import partial
import torch
import torch.nn as nn
from .utils import Block, LayerNorm
from timm.models.layers import trunc_normal_
from rsfm.backbone.swin.utils import load_checkpoint
from rsfm.module import build_ris_fusion, CIM, build_vg_fusion



class ConvNeXt(nn.Module):
    """
    A convnet for the 2020s, CVPR 2022
    """
    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 depths=[3, 3, 9, 3],
                 embed_dim=96,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3],
                 with_cp=False,
                 l_dim=768,
                 vlf_ris=None,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.,
                 vlf_vg=None):
        super().__init__()
        self.in_channels = in_channels
        self.depths = depths
        self.with_cp = with_cp
        embed_dims = [embed_dim * 2 ** i for i in range(len(out_indices))]

        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], kernel_size=4, stride=4),
            LayerNorm(embed_dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(embed_dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(embed_dims[i], embed_dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[Block(dim=embed_dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value, with_cp=self.with_cp) for j in range(depths[i])]
            )
            self.stages.append(stage)
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

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(len(depths)):
            layer = norm_layer(embed_dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, l=None, l_mask=None):
        B = x.shape[0]
        outs = []
        for i in range(len(self.depths)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

            if self.vlf_ris or self.vlf_vg:
                H, W = x.shape[2:]
                x = x.flatten(2).transpose(1, 2)
                fusion = getattr(self, f"fusion{i}")

                if self.vlf_ris == 'DMMI':
                    x, x_residual, l = fusion(x, l, l_mask)
                else:
                    x, x_residual = fusion(x, l, l_mask)

                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                x_residual = x_residual.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    x_out = norm_layer(x_residual)
                    outs.append(x_out)
            else:
                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    x_out = norm_layer(x)
                    outs.append(x_out)

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        if self.vlf_ris == 'DMMI':
            return l, tuple(outs)
        else:
            return tuple(outs)
