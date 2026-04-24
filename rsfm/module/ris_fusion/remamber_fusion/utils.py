import torch, math, einops
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from .vmamba_utils.vmamba_utils import SS2D, Linear2d, LayerNorm2d



class Twister(nn.Module):
    def __init__(self,
                 d_model=96,
                 d_state=16,
                 ssm_ratio=2.0,
                 dt_rank="auto",
                 act_layer=nn.SiLU,
                 d_conv=3,  # < 2 means no conv
                 conv_bias=True,
                 dropout=0.0,
                 bias=False,
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 initialize="v0",
                 forward_type="v2",
                 channel_first=True,
                 input_res=32,
                 **kwargs):
        super().__init__()

        self.input_res = input_res
        self.ss2d = SS2D(d_model, d_state, ssm_ratio, dt_rank, act_layer, d_conv, conv_bias, dropout, bias, dt_min,
                         dt_max, dt_init, dt_scale, dt_init_floor, initialize, forward_type, channel_first, **kwargs)
        self.ss2d.in_proj = Linear2d(d_model * 3, self.ss2d.in_proj.weight.shape[0], bias=bias)
        self.input_res = min(input_res, 16)
        forward_type_1d = "v052d"
        self.ss1d = SS2D(self.input_res ** 2, d_state, ssm_ratio, dt_rank, act_layer, d_conv, conv_bias, dropout, bias,
                         dt_min, dt_max, dt_init, dt_scale, dt_init_floor, initialize, forward_type_1d, channel_first,
                         **kwargs)

    def forward(self, x: torch.Tensor, **kwargs):
        img, global_cond, local_cond = x
        B, C, H, W = img.shape
        if global_cond is not None:
            x = torch.cat([img, global_cond, local_cond], dim=1)  # b l 3c
        else:
            x = torch.cat([img, local_cond], dim=-1)
        x_prepaired = x
        x_mix = F.interpolate(x, size=(self.input_res, self.input_res), mode='bilinear')
        x_mix = x_mix.view(B, -1, self.input_res ** 2)
        x_mix = x_mix.permute(0, 2, 1).unsqueeze(-2).contiguous()  # b, hw, 1, c
        x_mix = self.ss1d(x_mix)
        x_mix = x_mix.squeeze(-2).permute(0, 2, 1).contiguous()  # b, c, hw
        x_mix = F.interpolate(x_mix.view(B, -1, self.input_res, self.input_res), size=(H, W), mode='bilinear')

        x = x_mix + x_prepaired

        out = self.ss2d(x)

        out = [out, global_cond, local_cond]

        return out


class VSSBlock(nn.Module):
    def __init__(self,
                 dim=128,
                 forward_coremm="Twister",
                 norm_layer=LayerNorm2d,
                 drop_path=0.,
                 ssm_d_state=16,
                 ssm_ratio=2.0,
                 ssm_dt_rank="auto",
                 ssm_act_layer=nn.SiLU,
                 ssm_conv=3,
                 ssm_conv_bias=True,
                 ssm_drop_rate=0.,
                 ssm_init="v0",
                 forward_type="v2",
                 channel_first=True):
        super().__init__()

        self.ln_1 = norm_layer(dim)

        if forward_coremm == 'Twister':
            self.self_attention = Twister(
                d_model=dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )
        elif forward_coremm == 'SS2D':
            self.self_attention = SS2D(
                d_model=dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
                bias=False,
                dt_min=0.001,
                dt_max=0.1,
                dt_init="random",
                dt_scale=1.0,
                dt_init_floor=1e-4,
            )
        else:
            raise NotImplementedError

        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        if isinstance(input, torch.Tensor):
            out = self.ln_1(input)
            out = self.self_attention(out)
            out = input + self.drop_path(out)
            x = out
        else:
            # input should be a list (img and global / local conditions)
            out = [self.ln_1(i) if i is not None else None for i in input]
            out = self.self_attention(out)
            out = [i + self.drop_path(o) if i is not None else None for i, o in zip(input, out)]
            x = out

        return x


class VSSLayer(nn.Module):
    def __init__(self, dim=128, depth=2, downsample=None, dim_out=None, forward_coremm='Twister'):
        super().__init__()

        self.blocks = nn.ModuleList([VSSBlock(dim=dim, forward_coremm=forward_coremm) for _ in range(depth)])

        def _init_weights(module: nn.Module):
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    p = p.clone().detach_()  # fake init, just to keep the seed ....
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))

        self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, dim_out=dim_out, norm_layer=LayerNorm2d, channel_first=True)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        inner = x
        if self.downsample is not None:
            x = self.downsample(x)

        return x, inner


class ImageTextCorr(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim, out_dim, dropout=0.) -> None:
        super().__init__()

        self.vis_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.unsqueeze = nn.Sequential(
            nn.Conv2d(20, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, l_feat, l_mask):
        vis = self.vis_proj(einops.rearrange(x, 'b c h w -> b h w c'))
        txt = self.text_proj(einops.rearrange(l_feat, 'b c l -> b l c'))

        cost = torch.einsum("bhwc,blc->bhwl", vis, txt)  # s=h*w
        cost = einops.rearrange(cost, 'b h w c -> b c h w')

        feat = self.unsqueeze(cost)
        feat = einops.rearrange(feat, 'b c h w -> b h w c')

        out = self.out_proj(feat)

        return out
