import os, cv2
import mmcv
import numpy as np
from PIL import Image
from rsfm.dataset import datasets_info


path_label = '/path/to/your/local/dir'

dataset_type = 'DeepGlobe'

assert dataset_type in ['DeepGlobe', 'LoveDA', 'WHU_OPT_SAR', 'DFC24_T1', 'Agriculture_Vision', 'SPARCS', 'SegMunich']

cmap = datasets_info[dataset_type]['color_map']
save_label_path = path_label + '_visual'
os.makedirs(save_label_path, exist_ok=True)
names = os.listdir(path_label)

def wcm_visual(name):
    mask = np.array(Image.open(path_label + '/' + name))
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls in np.unique(mask):
        if cls != 255:
            mask_rgb[mask == cls] = cmap[cls]
        else:
            mask_rgb[mask == 255] = [0, 0, 0]
    cv2.imwrite(save_label_path + '/' + name, mask_rgb[:, :, ::-1])

mmcv.track_parallel_progress(wcm_visual, names, 32)



