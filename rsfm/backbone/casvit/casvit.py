import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from mmcv.runner import load_state_dict
from .utils import stem, Stage, Embedding
from rsfm.module import build_ris_fusion, build_vg_fusion, CIM



class CASViT(nn.Module):
    """
    CAS-ViT: Convolutional Additive Self-attention Vision Transformers for Efficient Mobile Applications
    """
    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 layers=[2, 2, 4, 2],
                 embed_dims=[48, 56, 112, 220],
                 mlp_ratios=4,
                 downsamples=[True, True, True, True],
                 norm_layer=nn.BatchNorm2d,
                 attn_bias=False,
                 act_layer=nn.GELU,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 l_dim=768,
                 vlf_ris=None,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.,
                 vlf_vg=None):
        super().__init__()
        self.layers = layers

        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg

        self.patch_embed = stem(in_channels, embed_dims[0])

        for i in range(len(layers)):
            stage = Stage(embed_dims[i], i, layers, mlp_ratio=mlp_ratios, act_layer=act_layer,
                          attn_bias=attn_bias, drop=drop_rate, drop_path_rate=drop_path_rate)
            setattr(self, f"stage{i}", stage)

            if self.vlf_ris:
                fusion = build_ris_fusion(vlf_ris, embed_dims[i], l_dim, num_heads_fusion[i], fusion_drop,
                                          size=img_size // (2 ** (i + 2)))
                setattr(self, f"fusion{i}", fusion)

            if self.vlf_vg:
                fusion = build_vg_fusion(vlf_vg, embed_dims[i], l_dim, size=img_size // (2 ** (i + 2)))
                setattr(self, f"fusion{i}", fusion)

            norm = norm_layer(embed_dims[i])
            setattr(self, f"norm{i}", norm)

            if i >= len(layers) - 1:
                break
            if downsamples[i]:
                downsample = Embedding(patch_size=3, stride=2, padding=1, in_chans=embed_dims[i],
                                       embed_dim=embed_dims[i+1], norm_layer=nn.BatchNorm2d)
                setattr(self, f"downsample{i}", downsample)

        if self.vlf_ris == 'RMSIN':
            self.cim = CIM(dim=sum(embed_dims), channels=embed_dims, height=img_size // 32, width=img_size // 32)

        self.init_weights()

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
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
        x = self.patch_embed(x)

        B = x.shape[0]
        outs = []
        for i in range(len(self.layers)):
            stage = getattr(self, f"stage{i}")
            norm = getattr(self, f"norm{i}")
            x = stage(x)

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

                out = norm(x_residual)
                outs.append(out)
                if i < len(self.layers) - 1:
                    downsample = getattr(self, f"downsample{i}")
                    x = downsample(x)
            else:
                out = norm(x)
                outs.append(out)
                if i < len(self.layers) - 1:
                    downsample = getattr(self, f"downsample{i}")
                    x = downsample(x)

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        if self.vlf_ris == 'DMMI':
            return l, tuple(outs)
        else:
            return tuple(outs)
