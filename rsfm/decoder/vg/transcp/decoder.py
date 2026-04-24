from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from .prototype import PrototypeLearner
from ..transvg.vl_transformer import build_vl_transformer
from ..vltvg.decoder import MULTIHEAD_ATTNS
from rsfm.utils import get_clones, MLP


class BboxRegression(nn.Module):
    def __init__(self, cfg, num_vis_token=400, aux_loss=False):
        super().__init__()
        args = cfg.copy()
        self.aux_loss = aux_loss

        layer_type = args.pop('type')
        self.layer = _MODULES[layer_type](**args)

        self.norm = nn.LayerNorm(256)
        self.num_visu_token = num_vis_token
        num_total = self.num_visu_token + 1
        self.vl_transformer = build_vl_transformer(dropout_rate=0., return_intermediate=aux_loss)
        self.vl_pos_embed = nn.Embedding(num_total, 256)
        self.reg_token = nn.Embedding(1, 256)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_feat, img_key_padding_mask, pos_embed, word_feat, word_mask,
                projected_disentangled_lang, h, w):
        hw, bs, c = img_feat.shape
        # Visual Context Disentangling + Prototype Learning
        x_multi_modal = self.layer(img_feat, img_key_padding_mask, pos_embed,
                                   word_feat, word_mask, projected_disentangled_lang, h, w)

        # Box reggresion (without box prediction head)
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)

        vl_src = torch.cat([tgt_src, x_multi_modal], dim=0)
        vl_mask = torch.cat([tgt_mask, img_key_padding_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.vl_transformer(vl_src, vl_mask, vl_pos)  # 6x(1+L+N)xBxC

        hs = []
        for pred in output:
            hs.append(self.norm(pred))
        hs = torch.stack(hs)

        return hs


class VisualDenstanglingPrototype(nn.Module):
    def __init__(self, num_queries, query_dim,
                 return_intermediate=False,
                 extra_layer=None, num_extra_layers=1):
        super().__init__()

        args = extra_layer.copy()
        layer_type = args.pop('type')
        extra_encoder_layer = _MODULES[layer_type](**args)
        self.extra_encoder_layers = get_clones(extra_encoder_layer, num_extra_layers)

        self.return_intermediate = return_intermediate
        self.vis_query_embed = nn.Embedding(num_queries, query_dim)
        self.text_query_embed = nn.Embedding(num_queries, query_dim)

        # prototype discovery module
        self.prototypelearner = PrototypeLearner(num_tokens=2048, decay=0.4)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, img_feat, img_key_padding_mask=None, pos=None,
                word_feat=None, word_key_padding_mask=None, projected_disentangled_lang=None, h=20, w=20):
        hw, bs, c = img_feat.shape

        # Visual Context Disentangling (Encode discriminative features)
        for layer in self.extra_encoder_layers:
            img_feat = layer(img_feat, img_key_padding_mask, pos,
                             word_feat, word_key_padding_mask, None)

        img_feat_srcs = img_feat.chunk(2, dim=-1)
        dis_img = img_feat_srcs[1]
        ori_img = img_feat_srcs[0]

        # prototype embedding
        prototype_out = self.prototypelearner(dis_img, h, w)
        dis_img_vd = prototype_out["embedded_pt"]

        x_multi_modal = torch.tanh(dis_img_vd.permute(1, 2, 0).contiguous().view(bs, -1, w, h)) * torch.tanh(
            projected_disentangled_lang)
        x_multi_modal = x_multi_modal.view(bs, -1, hw).permute(2, 0, 1)
        return x_multi_modal


class DiscriminativeFeatEncLayer(nn.Module):
    def __init__(self, d_model, img2text_attn_args=None, img_query_with_pos=True,
                 discrimination_coef_settings=None):
        super().__init__()
        args = img2text_attn_args.copy()
        self.img2text_attn = MULTIHEAD_ATTNS[args.pop('type')](**args)
        self.img_query_with_pos = img_query_with_pos

        self.text_proj = MLP(**discrimination_coef_settings['text_proj'])
        self.img_proj = MLP(**discrimination_coef_settings['img_proj'])
        self.tf_pow = discrimination_coef_settings.get('pow')
        self.tf_scale = Parameter(torch.Tensor([discrimination_coef_settings.get('scale')]))
        self.tf_sigma = Parameter(torch.Tensor([discrimination_coef_settings.get('sigma')]))
        self.norm_img = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, img_feat, img_key_padding_mask, img_pos,
                word_feat, word_key_padding_mask, word_pos=None):
        orig_img_feat = img_feat

        # discrimination coeficient calculation
        img_query = img_feat + img_pos if self.img_query_with_pos else img_feat

        # shared semantic between vision and language, Eq. (3)
        F_s = self.img2text_attn(
            query=img_query, key=self.with_pos_embed(word_feat, word_pos),
            value=word_feat, key_padding_mask=word_key_padding_mask)[0]

        # it can be seen as the salient objects in the language features
        # (e.g. "white dog besides the cat", "dog" is the salient object word in the language)
        text_embed = self.text_proj(F_s)

        #
        img_embed = self.img_proj(img_feat)

        # Eq. (4)
        dis_coef = (F.normalize(img_embed, p=2, dim=-1) *
                    F.normalize(text_embed, p=2, dim=-1)).sum(dim=-1, keepdim=True)
        dis_coef = self.tf_scale * \
                   torch.exp(- (1 - dis_coef).pow(self.tf_pow)
                             / (2 * self.tf_sigma ** 2))

        # Eq. (5)
        fuse_img_feat = self.norm_img(img_feat) * dis_coef
        return torch.cat([orig_img_feat, fuse_img_feat], dim=-1)


_MODULES = {
    'VisualDenstanglingPrototype': VisualDenstanglingPrototype,
    'DiscriminativeFeatEncLayer': DiscriminativeFeatEncLayer,
}

decoder_cfg = dict(
    type='VisualDenstanglingPrototype',
    num_queries=1,
    query_dim=256,
    return_intermediate=True,
    num_extra_layers=1,
    extra_layer=dict(
        type='DiscriminativeFeatEncLayer',
        d_model=256,
        img_query_with_pos=False,
        img2text_attn_args=dict(
            type='MultiheadAttention',
            embed_dim=256, num_heads=8, dropout=0.1
        ),
        discrimination_coef_settings=dict(
            text_proj=dict(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1),
            img_proj=dict(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1),
            scale=1.0,
            sigma=0.5,
            pow=2.0,
        ),
    )
)


def build_bbox_regression(num_vis_token, aux_loss):
    return BboxRegression(decoder_cfg, num_vis_token, aux_loss)