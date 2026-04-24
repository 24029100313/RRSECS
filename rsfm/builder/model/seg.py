from .base import BaseEncoderDecoder
import torch.nn.functional as F
import torch



class seg_model_builder(BaseEncoderDecoder):
    def __init__(self, cfg):
        super(seg_model_builder, self).__init__(cfg)


    def base_forward(self, x):
        h, w = x.shape[-2:]
        feats = self.backbone(x)
        if self.neck_cfg:
            feats = self.neck(feats)

        if self.dec_cfg['type'] == 'Mask2FormerHead':
            out_dict = self.decoder(feats)
            if self.training:
                return out_dict
            else:
                sem_seg = self.semantic_postprocess(out_dict, h, w)
                return sem_seg

        if self.dec_cfg['type'] in ['ABCNetHead', 'BANetHead']:
            out = F.interpolate(self.decoder(feats, x), size=(h, w), mode='bilinear', align_corners=False)
        else:
            out = F.interpolate(self.decoder(feats), size=(h, w), mode='bilinear', align_corners=False)

        return out


    def tta_forward(self, x):
        h, w = x.shape[-2:]

        final_result = None

        for scale in [1.0, 1.125, 1.25, 1.375, 1.5]:
            cur_h, cur_w = int(h * scale), int(w * scale)
            cur_x = F.interpolate(x, size=(cur_h, cur_w), mode='bilinear', align_corners=False)

            out = F.softmax(self.base_forward(cur_x), dim=1)
            out = F.interpolate(out, (h, w), mode='bilinear', align_corners=False)
            final_result = out if final_result is None else (final_result + out)

            out = F.softmax(self.base_forward(cur_x.flip(3)), dim=1).flip(3)
            out = F.interpolate(out, (h, w), mode='bilinear', align_corners=False)
            final_result += out

        return final_result


    def semantic_postprocess(self, out_dict, h, w):

        def _semantic_inference(mask_cls, mask_pred):
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

        mask_cls_results = out_dict["pred_logits"]
        mask_pred_results = out_dict["pred_masks"]

        mask_pred_results = F.interpolate(mask_pred_results, size=(h, w),
                                          mode="bilinear", align_corners=False)

        processed_results = []
        for mask_cls, mask_pred in zip(mask_cls_results, mask_pred_results):
            r = _semantic_inference(mask_cls, mask_pred)
            processed_results.append(r)

        processed_results = torch.stack(processed_results)

        return processed_results