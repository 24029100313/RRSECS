import torch
import torch.nn.functional as F
from .base import BaseEncoderDecoder
from rsfm.utils import NestedTensor, build_position_encoding
from pytorch_pretrained_bert.modeling import BertModel
from transformers import RobertaModel
from rsfm.decoder.vg.transcp import build_LSTMBert
from rsfm.language import AutoTinyBertModel



# class vg_model_builder(BaseEncoderDecoder):
#     def __init__(self, cfg):
#         super(vg_model_builder, self).__init__(cfg)
#
#         self.pos_embed = build_position_encoding()
#
#         self.text_enc_type = cfg['model']['text_encoder'].get('type', 'bert-base-uncased')
#         self.text_encoder = self.init_text_enc(self.text_enc_type)
#
#         if self.dec_cfg.get('type') == 'VLTVGHead':
#             for v in self.text_encoder.pooler.parameters():
#                 v.requires_grad_(False)
#
#         if self.dec_cfg.get('type') == 'TransCPHead':
#             assert self.text_enc_type == 'bert-base-uncased'
#             self.text_encoder = build_LSTMBert(self.text_encoder)
#
#
#     def base_forward(self, x, x_mask, l, l_mask):
#         text_feat, text_mask = self.language_forward(self.text_enc_type, l, l_mask)
#
#         if self.dec_cfg.get('type') == 'QRNetHead':
#             feats = self.backbone(x, text_feat[:, 0])
#         else:
#             feats = self.backbone(x)
#
#         if self.neck_cfg:
#             if self.neck_cfg.get('type') == 'QMF':
#                 feats = self.neck(feats, text_feat[:, 0])
#             else:
#                 feats = self.neck(feats)
#
#         outs, pos = [], []
#         for feat in feats:
#             mask = F.interpolate(x_mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
#             out = NestedTensor(feat, mask)
#             outs.append(out)
#             pos.append(self.pos_embed(out).to(out.tensors.dtype))
#
#         res = self.decoder(outs, pos, text_feat, text_mask)
#
#         return res
#
#
#     def tta_forward(self, x, x_mask, l, l_mask):
#         pass
#
#
#     def forward(self, x, x_mask, l, l_mask, tta=False):
#         if not tta:
#             return self.base_forward(x, x_mask, l, l_mask)
#         else:
#             return self.tta_forward(x, x_mask, l, l_mask)
#
#
#     def init_text_enc(self, type):
#         pretrained = 'pretrained_weights/bert/' + type
#
#         if type in ['bert-base-uncased', 'bert-large-uncased']:
#             text_encoder = BertModel.from_pretrained(type)
#         elif type == 'roberta-base':
#             text_encoder = RobertaModel.from_pretrained(pretrained)
#         elif 'autotinybert' in type:
#             text_encoder = AutoTinyBertModel(pretrained)
#         else:
#             raise NotImplementedError
#
#         return text_encoder
#
#
#     def language_forward(self, type, l, l_mask):
#         # basic BERT
#         if type in ['bert-base-uncased', 'bert-large-uncased']:
#             if self.dec_cfg.get('type') in ['TransVGHead', 'LQVGHead', 'PseudoQHead', 'QRNetHead']:
#                 all_enc_layer, _ = self.text_encoder(l, token_type_ids=None, attention_mask=l_mask)
#                 text_feat = all_enc_layer[-1]
#                 text_mask = l_mask.to(torch.bool)
#                 text_mask = ~text_mask
#             elif self.dec_cfg.get('type') == 'VLTVGHead':
#                 all_enc_layer, _ = self.text_encoder(l, token_type_ids=None, attention_mask=l_mask)
#                 text_feat = torch.stack(all_enc_layer[-4:], 1).mean(1)
#                 text_mask = l_mask.to(torch.bool)
#                 text_mask = ~text_mask
#             elif self.dec_cfg.get('type') == 'TransCPHead':
#                 # text_feat: disentangled_lang, text_mask: bert_fea
#                 text_feat, text_mask = self.text_encoder(NestedTensor(l, l_mask))
#             else:
#                 raise NotImplementedError
#         # improved BERT
#         elif type == 'roberta-base':
#             text_feat, text_mask = [], []
#             for s_l, s_l_mask in zip(l, l_mask):
#                 encoded_text = self.text_encoder(**{'input_ids': s_l, 'attention_mask': s_l_mask})
#                 s_text_feat = encoded_text.last_hidden_state
#                 text_feat.append(s_text_feat.squeeze(0))
#                 s_text_mask = s_l_mask.ne(1).bool()
#                 text_mask.append(s_text_mask.squeeze(0))
#             text_feat = torch.stack(text_feat)
#             text_mask = torch.stack(text_mask)
#         # lightweight BERT
#         elif 'autotinybert' in type:
#             text_feat = self.text_encoder(l, attention_mask=l_mask, kd=False)[0]
#             text_mask = l_mask.to(torch.bool)
#             text_mask = ~text_mask
#         else:
#             raise NotImplementedError
#
#         return text_feat, text_mask



class vg_model_builder(BaseEncoderDecoder):
    def __init__(self, cfg):
        super(vg_model_builder, self).__init__(cfg)

        self.pos_embed = build_position_encoding()

        self.text_enc_type = cfg['model']['text_encoder'].get('type', 'bert-base-uncased')
        self.text_encoder = self.init_text_enc(self.text_enc_type)

        if self.dec_cfg.get('type') == 'VLTVGHead':
            for v in self.text_encoder.pooler.parameters():
                v.requires_grad_(False)

        if self.dec_cfg.get('type') == 'TransCPHead':
            assert self.text_enc_type == 'bert-base-uncased'
            self.text_encoder = build_LSTMBert(self.text_encoder)


    def base_forward(self, x, x_mask, l, l_mask):
        text_feat, text_mask = self.language_forward(self.text_enc_type, l, l_mask)

        if self.dec_cfg.get('type') == 'QRNetHead':
            feats = self.backbone(x, text_feat[:, 0])
        elif self.dec_cfg.get('type') == 'LPVAHead':
            feats = self.backbone(x, text_feat)
        else:
            feats = self.backbone(x)

        if self.neck_cfg:
            neck_type = self.neck_cfg.get('type')
            if neck_type == 'QMF':
                feats = self.neck(feats, text_feat[:, 0])
            elif neck_type == 'vg_neck':
                feats = self.neck(feats, text_feat, text_mask)
            else:
                feats = self.neck(feats)

        outs, pos = [], []
        for feat in feats:
            mask = F.interpolate(x_mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out = NestedTensor(feat, mask)
            outs.append(out)
            pos.append(self.pos_embed(out).to(out.tensors.dtype))

        res = self.decoder(outs, pos, text_feat, text_mask)
        # res = self.decoder(outs, pos)

        return res


    def tta_forward(self, x, x_mask, l, l_mask):
        pass


    def forward(self, x, x_mask, l, l_mask, tta=False):
        if not tta:
            return self.base_forward(x, x_mask, l, l_mask)
        else:
            return self.tta_forward(x, x_mask, l, l_mask)


    def init_text_enc(self, type):
        pretrained = 'pretrained_weights/bert/' + type

        if type in ['bert-base-uncased', 'bert-large-uncased']:
            text_encoder = BertModel.from_pretrained(type)
        elif type == 'roberta-base':
            text_encoder = RobertaModel.from_pretrained(pretrained)
        elif 'autotinybert' in type:
            text_encoder = AutoTinyBertModel(pretrained)
        else:
            raise NotImplementedError

        return text_encoder


    def language_forward(self, type, l, l_mask):
        # basic BERT
        if type in ['bert-base-uncased', 'bert-large-uncased']:
            if self.dec_cfg.get('type') == 'VLTVGHead':
                all_enc_layer, _ = self.text_encoder(l, token_type_ids=None, attention_mask=l_mask)
                text_feat = torch.stack(all_enc_layer[-4:], 1).mean(1)
                text_mask = l_mask.to(torch.bool)
                text_mask = ~text_mask
            elif self.dec_cfg.get('type') == 'TransCPHead':
                # text_feat: disentangled_lang, text_mask: bert_fea
                text_feat, text_mask = self.text_encoder(NestedTensor(l, l_mask))
            else:
                all_enc_layer, _ = self.text_encoder(l, token_type_ids=None, attention_mask=l_mask)
                text_feat = all_enc_layer[-1]
                text_mask = l_mask.to(torch.bool)
                text_mask = ~text_mask
        # improved BERT
        elif type == 'roberta-base':
            text_feat, text_mask = [], []
            for s_l, s_l_mask in zip(l, l_mask):
                encoded_text = self.text_encoder(**{'input_ids': s_l, 'attention_mask': s_l_mask})
                s_text_feat = encoded_text.last_hidden_state
                text_feat.append(s_text_feat.squeeze(0))
                s_text_mask = s_l_mask.ne(1).bool()
                text_mask.append(s_text_mask.squeeze(0))
            text_feat = torch.stack(text_feat)
            text_mask = torch.stack(text_mask)
        # lightweight BERT
        elif 'autotinybert' in type:
            text_feat = self.text_encoder(l, attention_mask=l_mask, kd=False)[0]
            text_mask = l_mask.to(torch.bool)
            text_mask = ~text_mask
        else:
            raise NotImplementedError

        return text_feat, text_mask