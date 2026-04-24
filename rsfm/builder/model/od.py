import torch
import torch.nn.functional as F
from .base import BaseEncoderDecoder
from rsfm.utils import build_position_encoding, NestedTensor



class od_model_builder(BaseEncoderDecoder):
    def __init__(self, cfg):
        super(od_model_builder, self).__init__(cfg)

        self.pos_embed = build_position_encoding(position_embedding='sinehw')


    def base_forward(self, samples, targets):
        feats = self.backbone(samples.tensors)
        if self.neck_cfg:
            feats = self.neck(feats)

        features, poss = [], []
        for feat in feats:
            m = samples.mask
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out = NestedTensor(feat, mask)
            features.append(out)
            poss.append(self.pos_embed(out).to(out.tensors.dtype))

        if 'DINO' in self.dec_cfg.get('type', 'DETRHead'):
            out = self.decoder(features, poss, targets)
        else:
            out = self.decoder(features, poss)

        return out


    def tta_forward(self, samples, targets):
        # Todo: online weighted boxes fusion; Now, use offline weighted boxes fusion instead !!!
        pass


    def forward(self, samples, targets=None, tta=False):
        if not tta:
            return self.base_forward(samples, targets)
        else:
            return self.tta_forward(samples, targets)