from .base import BaseEncoderDecoder
import importlib
import torch.nn.functional as F
from rsfm.dataset.utils import datasets_info
from rsfm.builder.pretrained_info import get_pretrained_path


class cls_model_builder(BaseEncoderDecoder):
    def __init__(self, cfg):
        super(cls_model_builder, self).__init__(cfg)


    def _build_backbone(self):
        self.enc_cfg['kwargs']['img_size'] = self.cfg['crop_size']
        self.enc_cfg['kwargs']['num_classes'] = datasets_info[self.cfg['dataset']]['num_classes']

        encoder = self._build_enc_module(self.enc_cfg['type'], self.enc_cfg['kwargs'])
        encoder.init_weights(get_pretrained_path(self.enc_cfg.get('pretrained'), self.enc_cfg['type']))

        return encoder


    def _build_enc_module(self, mtype, kwargs):
        enc = getattr(importlib.import_module('rsfm.classification'), mtype)
        return enc(**kwargs)


    def base_forward(self, x):
        out = self.backbone(x)
        return out


    def tta_forward(self, x):
        h, w = x.shape[-2:]

        final_result = None

        for scale in [1.0, 1.125, 1.25, 1.375, 1.5]:
            cur_h, cur_w = int(h * scale), int(w * scale)
            cur_x = F.interpolate(x, size=(cur_h, cur_w), mode='bilinear', align_corners=False)

            out = F.softmax(self.base_forward(cur_x), dim=1)
            final_result = out if final_result is None else (final_result + out)

            out = F.softmax(self.base_forward(cur_x.flip(3)), dim=1).flip(3)
            final_result += out

        return final_result