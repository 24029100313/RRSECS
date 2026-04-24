import json
import shutil
import cv2
import imagesize, os
import mmcv
import numpy as np
from tqdm import tqdm


save_path = '/root/data2/comp_masks/refcoco/7_magnet2vg'
os.makedirs(save_path, exist_ok=True)


def mask2bbox(path):
    # ratio = 800 / 512
    ratio = 1

    ris2vg_dict = {}
    for file in os.listdir(path):
        mask = cv2.imread(path + '/' + file, 0)
        conts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if conts:
            all_points = np.vstack(conts)
            x, y, w, h = cv2.boundingRect(all_points)
            bbox = [x * ratio, y * ratio, (x + w) * ratio, (y + h) * ratio]
            ris2vg_dict[file.split('.')[0]] = bbox
        else:
            ris2vg_dict[file.split('.')[0]] = [0, 0, 0, 0]

    json.dump(ris2vg_dict, open(save_path + '/' + os.path.basename(path) + '2vg.json', 'w'))


def mask2bbox_mo(path):
    # ratio = 800 / 512
    ratio = 1

    ris2vg_dict = {}
    for file in tqdm(os.listdir(path)):
        mask = cv2.imread(path + '/' + file, 0)

        conts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if conts:
            simo = []
            for cont in conts:
                x, y, w, h = cv2.boundingRect(cont)
                simo.append(([x, y, w, h], w * h))
            max_o = sorted(simo, key=lambda x: x[1], reverse=True)[0][0]
            bbox = [max_o[0] * ratio,
                    max_o[1] * ratio,
                    (max_o[0] + max_o[2]) * ratio,
                    (max_o[1] + max_o[3]) * ratio]
            ris2vg_dict[file.split('.')[0]] = bbox
        else:
            ris2vg_dict[file.split('.')[0]] = [0, 0, 0, 0]

    json.dump(ris2vg_dict, open(save_path + '/' + os.path.basename(path) + '2vg.json', 'w'))


# path = '/root/lxq/RefDIOR_results/RRSIS/wosa/test/masks'
# dirs = []
# for model in os.listdir(path):
#     dirs.append(path + '/' + model)
#
# mmcv.track_parallel_progress(mask2bbox_mo, dirs, 24)

mask2bbox('/root/data2/comp_masks/refcoco/7_magnet')