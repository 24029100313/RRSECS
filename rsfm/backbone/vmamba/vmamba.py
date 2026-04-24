import torch
import torch.nn as nn
from collections import OrderedDict
from mmcv.runner import load_state_dict
from timm.models.layers import trunc_normal_
from rsfm.module.ris_fusion import build_ris_fusion, CIM
from rsfm.module.vg_fusion import build_vg_fusion
from .utils import VSSBlock, Permute



class VSSM(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=4,
            in_channels=3,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",  # "BN", "LN2D"
            downsample_version: str = "v3",  # "v1", "v2", "v3"
            patchembed_version: str = "v2",  # "v1", "v2"
            with_cp=False,
            posembed=False,
            out_indices=(0, 1, 2, 3),
            l_dim=768,
            vlf_ris=None,
            num_heads_fusion=[1, 1, 1, 1],
            fusion_drop=0.,
            vlf_vg=None,
            **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_layers = len(depths)

        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg

        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, img_size) if posembed else None

        _make_patch_embed = dict(
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_channels, dims[0], patch_size, patch_norm,
                                             norm_layer)  # , channel_first=self.channel_first)

        _make_downsample = dict(
            v3=self._make_downsample_v3,
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=norm_layer,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                with_cp=with_cp,
                norm_layer=norm_layer,
                downsample=downsample,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                l_dim=l_dim,
                vlf_ris=self.vlf_ris,
                num_heads_fusion=num_heads_fusion[i_layer],
                fusion_drop=fusion_drop,
                vlf_vg=self.vlf_vg,
                size=img_size // (2 ** (i_layer + 2))
            ))
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        if self.vlf_ris == 'RMSIN':
            self.cim = CIM(dim=sum(dims), channels=dims, height=img_size // 32, width=img_size // 32)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def init_weights(self, pretrained=None):
        def _init_weights(m: nn.Module):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            ckpt = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            load_state_dict(self, _state_dict, False)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    @staticmethod
    def _make_patch_embed_v2(in_channels=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                             channel_first=False):
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            with_cp=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            channel_first=False,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            l_dim=768,
            vlf_ris=None,
            num_heads_fusion=1,
            fusion_drop=0.,
            vlf_vg=None,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                with_cp=with_cp,
            ))
        if vlf_ris:
            fusion = build_ris_fusion(vlf_ris, dim, l_dim, num_heads_fusion, fusion_drop, **kwargs)

            return nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*blocks, ),
                fusion=fusion,
                downsample=downsample,
            ))

        if vlf_vg:
            fusion = build_vg_fusion(vlf_vg, dim, l_dim, **kwargs)

            return nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*blocks, ),
                fusion=fusion,
                downsample=downsample,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))

    def forward(self, x, l=None, l_mask=None):
        def layer_forward(layer, x, l, l_mask):
            x = layer.blocks(x)
            if self.vlf_ris or self.vlf_vg:
                B, H, W, C = x.shape
                x = x.permute(0, 3, 1, 2).flatten(2).transpose(1, 2)

                if self.vlf_ris == 'DMMI':
                    x, x_residual, l = layer.fusion(x, l, l_mask)
                else:
                    x, x_residual = layer.fusion(x, l, l_mask)

                x, x_residual = x.reshape(B, H, W, -1), x_residual.reshape(B, H, W, -1)
                y = layer.downsample(x)

                if self.vlf_ris == 'DMMI':
                    return x_residual, y, l
                else:
                    return x_residual, y
            else:
                y = layer.downsample(x)
                return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):

            if self.vlf_ris == 'DMMI':
                o, x, l = layer_forward(layer, x, l, l_mask)  # (B, H, W, C)
            else:
                o, x = layer_forward(layer, x, l, l_mask)  # (B, H, W, C)

            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        if self.vlf_ris == 'DMMI':
            return l, outs
        else:
            return outs