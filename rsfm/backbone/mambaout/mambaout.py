import torch
import torch.nn as nn
from functools import partial
from timm.layers import trunc_normal_
from mmcv.runner import load_state_dict
from .utils import GatedCNNBlock, StemLayer, DownsampleLayer
from rsfm.module import build_ris_fusion, CIM, build_vg_fusion



class MambaOut(nn.Module):
    '''
    MambaOut: Do We Really Need Mamba for Vision?
    '''
    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 depths=[3, 3, 9, 3],
                 embed_dims=[96, 192, 384, 576],
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 conv_ratio=1.0,
                 kernel_size=7,
                 drop_path_rate=0.,
                 l_dim=768,
                 vlf_ris=None,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.,
                 vlf_vg=None,
                 **kwargs):
        super().__init__()

        num_stage = len(depths)
        self.num_stage = num_stage

        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg

        downsample_layers = [StemLayer, DownsampleLayer, DownsampleLayer, DownsampleLayer]

        down_dims = [in_channels] + embed_dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i+1]) for i in range(num_stage)]
        )

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[GatedCNNBlock(
                    dim=embed_dims[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    kernel_size=kernel_size,
                    conv_ratio=conv_ratio,
                    drop_path=dp_rates[cur + j]
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

            norm = norm_layer(embed_dims[i])
            setattr(self, f"norm{i}", norm)

            if self.vlf_ris:
                fusion = build_ris_fusion(vlf_ris, embed_dims[i], l_dim, num_heads_fusion[i], fusion_drop,
                                          size=img_size // (2 ** (i + 2)))
                setattr(self, f"fusion{i}", fusion)

            if self.vlf_vg:
                fusion = build_vg_fusion(vlf_vg, embed_dims[i], l_dim, size=img_size // (2 ** (i + 2)))
                setattr(self, f"fusion{i}", fusion)

        if self.vlf_ris == 'RMSIN':
            self.cim = CIM(dim=sum(embed_dims), channels=embed_dims, height=img_size // 32, width=img_size // 32)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            checkpoint = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            load_state_dict(self, state_dict, False)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, l=None, l_mask=None):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x) # B, H, W, C

            if self.vlf_ris or self.vlf_vg:
                x = x.permute(0, 3, 1, 2)
                H, W = x.shape[2:]
                x = x.flatten(2).transpose(1, 2)
                fusion = getattr(self, f"fusion{i}")

                if self.vlf_ris == 'DMMI':
                    x, x_residual, l = fusion(x, l, l_mask)
                else:
                    x, x_residual = fusion(x, l, l_mask)

                x = x.reshape(B, H, W, -1).contiguous()
                x_residual = x_residual.reshape(B, H, W, -1).contiguous()

                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_residual)
                outs.append(x_out.permute(0, 3, 1, 2))

            else:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out.permute(0, 3, 1, 2))

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        if self.vlf_ris == 'DMMI':
            return l, tuple(outs)
        else:
            return tuple(outs)