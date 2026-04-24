import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from rsfm.foundation_model.sam.v1 import SamPredictor, sam_model_registry


def get_sam_prompt_assistor(model_type='vit_b', device='cuda'):
    sam = sam_model_registry[model_type](checkpoint='pretrained/sam/sam_' + model_type + '.pth')
    sam.to(device=device)
    sam_prompt_assistor = SamPredictor(sam)

    return sam_prompt_assistor



def det_sam_mask(vg_json, img_path, save_path, img_suffix='.jpg', type='vit_b'):
    os.makedirs(save_path, exist_ok=True)

    sam_prompt_assistor = get_sam_prompt_assistor(type)
    preds = json.load(open(vg_json))
    for k, v in tqdm(preds.items()):
        img = np.array(Image.open(img_path + '/' + k + img_suffix))
        sam_prompt_assistor.set_image(img)
        box_coord_prompt = np.array(v)
        masks, _, _ = sam_prompt_assistor.predict(point_coords=None,
                                                  point_labels=None,
                                                  box=box_coord_prompt[None,:],
                                                  multimask_output=False)
        cv2.imwrite(save_path + '/' + k + '.png', (255 * masks[0]).astype(np.uint8))


det_sam_mask(vg_json='/root/lxq/RefDIOR_results/RSVG/wosa/test/jsons/RefDIOR_VG.swin_tiny.None.LQVGHead.4xb8.img512.ep50.preimagenet.best81.86_predictions.json',
             img_path='/root/lxq/RSFM_RIS_Datasets/Optical/RefDIOR/images',
             save_path='/root/lxq/RefDIOR_results/RSVG/wosa/test/masks/RefDIOR_VG.swin_tiny.None.LQVGHead.4xb8.img512.ep50.preimagenet.best81.86_predictions/sam_vit_b')