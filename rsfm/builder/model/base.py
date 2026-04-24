import importlib
from torch import nn
from rsfm.builder.model_info import *
from rsfm.builder.pretrained_info import get_pretrained_path
from rsfm.dataset.utils import datasets_info



class BaseEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super(BaseEncoderDecoder, self).__init__()
        self.cfg = cfg
        self.enc_cfg, self.neck_cfg, self.dec_cfg = self._base_init(cfg)

        self.backbone = self._build_backbone()

        if self.neck_cfg:
            self.neck = self._build_neck()

        if self.dec_cfg:
            self.decoder = self._build_decoder()


    def _base_init(self, cfg):
        model_cfg = cfg['model']

        enc_cfg = model_cfg.get('backbone')
        neck_cfg = model_cfg.get('neck')
        dec_cfg = model_cfg.get('decoder')

        return enc_cfg, neck_cfg, dec_cfg


    def _build_backbone(self):
        enc_type, enc_kwargs = self.enc_cfg.get('type'), self.enc_cfg.get('kwargs', {})

        enc_kwargs['img_size'] = self.cfg.get('crop_size', datasets_info[self.cfg['dataset']]['training_size'])

        if self.cfg['model'].get('text_encoder', False):
            enc_kwargs['l_dim'] = language_info[self.cfg['model']['text_encoder']['type']]['hidden_dim']

        encoder = self._build_enc_module(enc_type, enc_kwargs)
        encoder.init_weights(get_pretrained_path(self.enc_cfg.get('pretrained'), self.enc_cfg['type']))

        return encoder


    def _build_neck(self):
        self.neck_cfg['kwargs'] = {}
        self.neck_cfg['kwargs']['in_channels'] = encoder_info[self.enc_cfg['type']]['enc_out_dims']

        if self.neck_cfg['type'] == 'p2h':
            self.neck_cfg['kwargs']['out_channels'] = [self.neck_cfg['kwargs']['in_channels'][-1] // 2 ** (i - 1)
                                                       for i in range(4, 0, -1)]
        elif self.neck_cfg['type'] == 'multilevel_neck':
            self.neck_cfg['kwargs']['out_channels'] = self.neck_cfg['kwargs']['in_channels'][-1]
        else:
            self.neck_cfg['kwargs']['out_channels'] = neck_info[self.neck_cfg['type']]['out_channels']

        neck = self._build_neck_module(self.neck_cfg['type'], self.neck_cfg['kwargs'])
        return neck


    def _build_decoder(self):
        dec_type, dec_kwargs = self.dec_cfg.get('type'), self.dec_cfg.get('kwargs', {})
        assert dec_type != None, 'What are you doing ? You must set the decoder type !!!'

        dec_kwargs['embedding_dim'] = decoder_info[dec_type]['dec_out_dim']
        if dec_type in ['SegFormerHead', 'UMixFormerHead']:
            if self.enc_cfg['type'].split('_')[1] in ['b2', 'b3', 'b4', 'b5']:
                dec_kwargs['embedding_dim'] = 512

        dec_kwargs['num_classes'] = dec_kwargs.get('num_classes', datasets_info[self.cfg['dataset']]['num_classes'])

        dec_kwargs['in_channels'] = encoder_info[self.enc_cfg['type']]['enc_out_dims']
        if self.neck_cfg:
            dec_kwargs['in_channels'] = self.neck_cfg['kwargs']['out_channels']
            if isinstance(dec_kwargs['in_channels'], int):
                dec_kwargs['in_channels'] = [dec_kwargs['in_channels']] * 4

        dec_kwargs['img_size'] = self.cfg.get('crop_size', datasets_info[self.cfg['dataset']]['training_size'])

        if self.cfg['criterion']['kwargs'].get('aux_loss', False):
            dec_kwargs['aux_loss'] = True

        decoder = self._build_dec_module(dec_type, dec_kwargs)
        return decoder


    def _build_enc_module(self, mtype, kwargs):
        enc = getattr(importlib.import_module('rsfm.backbone'), mtype)
        return enc(**kwargs)


    def _build_neck_module(self, mtype, kwargs):
        neck = getattr(importlib.import_module('rsfm.module'), mtype)
        return neck(**kwargs)


    def _build_dec_module(self, mtype, kwargs):
        dec = getattr(importlib.import_module('rsfm.decoder'), mtype)
        return dec(**kwargs)


    def base_forward(self, x):
        pass


    def tta_forward(self, x):
        pass


    def forward(self, x, tta=False):
        if not tta:
            return self.base_forward(x)
        else:
            return self.tta_forward(x)