import os
from tqdm import tqdm
import numpy as np
import cv2


pred_path = '/root/lxq/RSFMv0.3.2/exps/ris/QaTa-COV19/pvtv2_b0_lavt/QaTa-COV19.pvtv2_b0.None.LAVTHead.4xb8.img224.ep50.preimagenet.best94.51_predictions'
pred_names = os.listdir(pred_path)
test_num = len(pred_names)
gt_path = '/root/lxq/RSFM_Seg_Datasets/Medical/QaTa-COV19-Seg/val/labels'


def metric_res(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection

    dice = 2 * intersection / (np.sum(y_true) + np.sum(y_pred) + smooth)
    iou = intersection / (union + smooth)

    return dice, iou


miou, dice = 0, 0
for pred_name in tqdm(pred_names):
    pred = cv2.imread(pred_path + '/' + pred_name, 0)
    pred[(pred == 255)] = 1
    gt = cv2.imread(gt_path + '/' + pred_name, 0)
    gt[(gt == 255)] = 1

    infection_dice, infection_iou = metric_res(gt, pred)

    miou += infection_iou
    dice += infection_dice

print('dice:', round(100 * dice / test_num, 2))
print('miou:', round(100 * miou / test_num, 2))
