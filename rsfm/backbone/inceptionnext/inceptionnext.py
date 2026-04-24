import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .utils import InceptionDWConv2d, MetaNeXtStage
from mmcv.runner import load_state_dict
from rsfm.module import build_ris_fusion, CIM, build_vg_fusion


class MetaNeXt(nn.Module):
    """
    InceptionNeXt: When Inception Meets ConvNeXt, CVPR 2024
    """
    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 depths=(3, 3, 9, 3),
                 embed_dims=(96, 192, 384, 768),
                 token_mixers=InceptionDWConv2d,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.GELU,
                 mlp_ratios=(4, 4, 4, 3),
                 drop_rate=0.,
                 drop_path_rate=0.,
                 ls_init_value=1e-6,
                 l_dim=768,
                 vlf_ris=None,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.,
                 vlf_vg=None):
        super().__init__()

        self.depths = depths

        num_stage = len(depths)
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage
        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * num_stage

        self.drop_rate = drop_rate
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], kernel_size=4, stride=4),
            norm_layer(embed_dims[0])
        )

        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg

        self.stages = nn.ModuleList()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        for i in range(num_stage):
            self.stages.append(
                MetaNeXtStage(
                    in_chs=embed_dims[i - 1] if i > 0 else embed_dims[0],
                    out_chs=embed_dims[i],
                    ds_stride=2 if i > 0 else 1,
                    depth=depths[i],
                    drop_path_rates=dp_rates[i],
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    token_mixer=token_mixers[i],
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratios[i]))

            if self.vlf_ris:
                fusion = build_ris_fusion(vlf_ris, embed_dims[i], l_dim, num_heads_fusion[i],
                                          fusion_drop, size=img_size // (2 ** (i + 2)))
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
        x = self.stem(x)

        for i in range(len(self.depths)):
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
                outs.append(x_residual)
            else:
                outs.append(x)

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        if self.vlf_ris == 'DMMI':
            return l, tuple(outs)
        else:
            return tuple(outs)