import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .utils import ConvBN, Block
from mmcv.runner import load_state_dict
from rsfm.module import build_ris_fusion, CIM, build_vg_fusion



class StarNet(nn.Module):
    '''
    Rewrite the Stars, CVPR 2024
    '''
    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 base_dim=32,
                 depths=[3, 3, 12, 5],
                 mlp_ratio=4,
                 drop_path_rate=0.0,
                 l_dim=768,
                 vlf_ris=None,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.,
                 vlf_vg=None,
                 **kwargs):
        super().__init__()
        self.in_channel = 32
        self.depths = depths

        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg

        # stem layer
        self.stem = nn.Sequential(ConvBN(in_channels, self.in_channel, kernel_size=3, stride=2, padding=1),
                                  nn.ReLU6())

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        cur = 0

        # build stages
        embed_dims =  []
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            embed_dim = base_dim * 2 ** i
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + j]) for j in range(depths[i])]
            cur += depths[i]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

            norm = nn.BatchNorm2d(self.in_channel)
            setattr(self, f"norm{i}", norm)

            if self.vlf_ris:
                fusion = build_ris_fusion(vlf_ris, self.in_channel, l_dim, num_heads_fusion[i], fusion_drop,
                                          size=img_size // (2 ** (i + 2)))
                setattr(self, f"fusion{i}", fusion)

            if self.vlf_vg:
                fusion = build_vg_fusion(vlf_vg, self.in_channel, l_dim, size=img_size // (2 ** (i + 2)))
                setattr(self, f"fusion{i}", fusion)

            embed_dims.append(self.in_channel)

        if self.vlf_ris == 'RMSIN':
            self.cim = CIM(dim=sum(embed_dims), channels=embed_dims, height=img_size // 32, width=img_size // 32)


    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear or nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

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

                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_residual)
                outs.append(x_out)
            else:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        if self.vlf_ris == 'DMMI':
            return l, tuple(outs)
        else:
            return tuple(outs)

