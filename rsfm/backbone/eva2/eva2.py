import torch
from functools import partial
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from mmcv.runner import load_state_dict
from timm.models.layers import trunc_normal_
from rsfm.module import build_ris_fusion, CIM, build_vg_fusion
from .utils import HybridEmbed, PatchEmbed, RelativePositionBias, VisionRotaryEmbeddingFast, Block



class EVA2(nn.Module):
    """
    EVA-02: A Visual Representation for Neon Genesis, IVC 2024
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4 * 2 / 3,  # GLU default
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            hybrid_backbone=None,
            norm_layer=None,
            init_values=None,
            with_cp=False,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=False,
            out_indices=[3, 5, 7, 11],
            subln=True,
            xattn=False, # install xformers if True
            naiveswiglu=True,
            rope=True,
            pt_hw_seq_len=16,
            intp_freq=True,
            l_dim=768,
            vlf_ris=None,
            num_heads_fusion=[1, 1, 1, 1],
            fusion_drop=0.,
            vlf_vg=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_channels, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches
        self.out_indices = out_indices

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
            )
        else:
            self.rope = None

        self.naiveswiglu = naiveswiglu

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.with_cp = with_cp

        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  init_values=init_values,
                  window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                  subln=subln,
                  xattn=xattn,
                  naiveswiglu=naiveswiglu,
                  rope=self.rope)
            for i in range(depth)
        ])

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        if self.vlf_ris:
            for i, out_indice in enumerate(self.out_indices):
                fusion = build_ris_fusion(vlf_ris, embed_dim, l_dim, num_heads_fusion[i], fusion_drop,
                                          size=img_size // patch_size)
                setattr(self, f"fusion{out_indice}", fusion)

            if self.vlf_ris == 'RMSIN':
                self.cim = CIM(dim=embed_dim * 4,
                               channels=[embed_dim] * 4,
                               height=img_size // patch_size,
                               width=img_size // patch_size)

        if self.vlf_vg:
            for i, out_indice in enumerate(self.out_indices):
                fusion = build_vg_fusion(vlf_vg, embed_dim, l_dim, size=img_size // patch_size)
                setattr(self, f"fusion{out_indice}", fusion)

        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Identity()

            self.fpn3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
            )

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
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

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, l=None, l_mask=None):
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        outs = []
        for i, blk in enumerate(self.blocks):
            if self.with_cp:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)

            if i in self.out_indices:
                out = x[:, 1:, :]
                cls_tokens = x[:, 0:1, :]

                if self.vlf_ris or self.vlf_vg:
                    if self.vlf_ris == 'DMMI':
                        x, out, l = self.__getattr__(f"fusion{i}")(out, l, l_mask)
                    else:
                        x, out = self.__getattr__(f"fusion{i}")(out, l, l_mask)
                    x = torch.cat((cls_tokens, x), dim=1)

                out = out.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
                outs.append(out)

        if self.vlf_ris == 'RMSIN':
            outs = self.cim(outs)

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(outs)):
            outs[i] = ops[i](outs[i])

        if self.vlf_ris == 'DMMI':
            return l, tuple(outs)
        else:
            return tuple(outs)