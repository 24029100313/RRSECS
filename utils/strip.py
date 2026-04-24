import os
import torch
import mmcv



def strip_model(name):
    ckpt = torch.load(name, map_location='cpu')
    strip_ckpt = {'model': ckpt['model']}
    torch.save(strip_ckpt, name)

# strip_model('exps/vg/OPT-RSVG/internimage_huge_vltvg_8xb2_img640_ep50_b5g2_auxloss/OPT-RSVG.internimage_huge.None.VLTVGHead.8xb2.img640.ep50.preimagenet.best83.0.pth')

# mmcv.track_parallel_progress(strip_model, names, 8)
