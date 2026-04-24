import torch
import torch.nn.functional as F
from .base import BaseEncoderDecoder
from transformers import RobertaModel
from rsfm.language import BertModel, AutoTinyBertModel



class ris_model_builder(BaseEncoderDecoder):
    def __init__(self, cfg):
        super(ris_model_builder, self).__init__(cfg)

        self.text_enc_type = cfg['model']['text_encoder'].get('type', 'bert-base-uncased')
        self.text_encoder = self.init_text_enc(self.text_enc_type)

    def base_forward(self, x, l, l_mask):
        h, w = x.shape[-2:]

        l_feats, l_mask = self.language_forward(self.text_enc_type, l, l_mask)

        if self.enc_cfg['kwargs'].get('vlf_ris') == 'DMMI':
            l_feats, feats = self.backbone(x, l_feats, l_mask)
        else:
            feats = self.backbone(x, l_feats, l_mask)

        if self.neck_cfg:
            if self.neck_cfg.get('type') == 'LGFPN':
                feats = self.neck(feats, l_feats)
            else:
                feats = self.neck(feats)

        out = self.various_decoder(self.dec_cfg['type'], feats, l_feats, l_mask, h, w)

        return out


    def tta_forward(self, x, l, l_mask):
        pass


    def forward(self, x, l, l_mask, tta=False):
        if not tta:
            return self.base_forward(x, l, l_mask)
        else:
            return self.tta_forward(x, l, l_mask)


    def init_text_enc(self, type):
        pretrained = 'pretrained/bert/' + type

        if type in ['bert-base-uncased', 'bert-large-uncased']:
            text_encoder = BertModel.from_pretrained(pretrained)
            text_encoder.pooler = None
        elif type == 'roberta-base':
            text_encoder = RobertaModel.from_pretrained(pretrained)
            text_encoder.pooler = None
        elif 'autotinybert' in type:
            text_encoder = AutoTinyBertModel(pretrained)
            text_encoder.model.bert.pooler = None
            text_encoder.model.bert.dense_fit = None
        else:
            raise NotImplementedError

        return text_encoder


    def language_forward(self, type, l, l_mask):
        # basic BERT
        if type in ['bert-base-uncased', 'bert-large-uncased']:
            l_feats = self.text_encoder(l, attention_mask=l_mask)[0]  # (B, 20, 768)
            l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        # improved BERT
        elif type == 'roberta-base':
            text_feat, text_mask = [], []
            for s_l, s_l_mask in zip(l, l_mask):
                encoded_text = self.text_encoder(**{'input_ids': s_l, 'attention_mask': s_l_mask})
                s_text_feat = encoded_text.last_hidden_state
                text_feat.append(s_text_feat.squeeze(0))
                s_text_mask = s_l_mask.ne(1).bool()
                text_mask.append(s_text_mask.squeeze(0))
            l_feats = torch.stack(text_feat)
            l_mask = torch.stack(text_mask)
        # lightweight BERT
        elif 'autotinybert' in type:
            l_feats = self.text_encoder(l, attention_mask=l_mask, kd=False)[0]
            l_feats = l_feats.permute(0, 2, 1)
            l_mask = l_mask.unsqueeze(dim=-1)
        else:
            raise NotImplementedError

        return l_feats, l_mask


    def various_decoder(self, type, feats, l_feats, l_mask, h, w):

        if type in ['LGCEHead', 'ReLAHead', 'MCTHead', 'DMMIHead']:
            out = F.interpolate(self.decoder(feats, l_feats), size=(h, w), mode='bilinear', align_corners=False)

        elif type in ['CGFormerHead', 'ReMamberHead', 'CCFormerRISHead']:
            out = F.interpolate(self.decoder(feats, l_feats, l_mask), size=(h, w), mode='bilinear', align_corners=False)

        elif type == 'Mask2FormerHead':
            out = F.interpolate(self.decoder(feats)['pred_masks'], size=(h, w), mode='bilinear', align_corners=False)

        else:
            out = F.interpolate(self.decoder(feats), size=(h, w), mode='bilinear', align_corners=False)

        return out