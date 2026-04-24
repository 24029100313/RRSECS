import torch
import torch.nn.functional as F
from .base import BaseEncoderDecoder



class cd_model_builder(BaseEncoderDecoder):
    def __init__(self, cfg):
        super(cd_model_builder, self).__init__(cfg)

        if not self.enc_cfg.get('siamese', True):
            self.backbone1 = self._build_backbone()
            if self.neck_cfg:
                self.neck1 = self._build_neck()

    def base_forward(self, x1, x2):
        h, w = x1.shape[-2:]

        if not self.enc_cfg.get('siamese', True):
            feats1 = self.backbone(x1)
            feats2 = self.backbone1(x2)
            if self.neck_cfg:
                feats1 = self.neck(feats1)
                feats2 = self.neck1(feats2)
            feats = [torch.cat((f1, f2)) for f1, f2 in zip(feats1, feats2)]
        else:
            feats = self.backbone(torch.cat((x1, x2)))
            if self.neck_cfg:
                feats = self.neck(feats)

        # using cd-specific decoder head for interpretation
        if 'CD' in self.dec_cfg['type']:
            out = F.interpolate(self.decoder(feats), size=(h, w), mode='bilinear', align_corners=False)
            return out

        # using simple feature difference with seg decoder head for interpretation
        feats_differ = []
        for feat in feats:
            feat1, feat2 = feat.chunk(2)
            feats_differ.append(torch.abs(feat1 - feat2))

        out = F.interpolate(self.decoder(feats_differ), size=(h, w), mode='bilinear', align_corners=False)
        return out


    def tta_forward(self, x1, x2):
        h, w = x1.shape[-2:]

        final_result = None

        for scale in [1.0, 1.125, 1.25, 1.375, 1.5]:
            cur_h, cur_w = int(h * scale), int(w * scale)
            cur_x1 = F.interpolate(x1, size=(cur_h, cur_w), mode='bilinear', align_corners=False)
            cur_x2 = F.interpolate(x2, size=(cur_h, cur_w), mode='bilinear', align_corners=False)

            out = F.softmax(self.base_forward(cur_x1, cur_x2), dim=1)
            out = F.interpolate(out, (h, w), mode='bilinear', align_corners=False)
            final_result = out if final_result is None else (final_result + out)

            out = F.softmax(self.base_forward(cur_x1.flip(3), cur_x2.flip(3)), dim=1).flip(3)
            out = F.interpolate(out, (h, w), mode='bilinear', align_corners=False)
            final_result += out

        return final_result

    def forward(self, x1, x2, tta=False):
        if not tta:
            return self.base_forward(x1, x2)
        else:
            return self.tta_forward(x1, x2)