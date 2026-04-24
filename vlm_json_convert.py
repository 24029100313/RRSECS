import json, re
import os.path
from tqdm import tqdm
import numpy as np
from utils.vg_off_eval import bbox_iou, metric_table


# iou_5, iou_6, iou_7, iou_8, iou_9, ious, inter, union = 0., 0., 0., 0., 0., [], 0., 0.
#
# gts = json.load(open('/root/lxq/RSFM_RIS_Datasets/Optical/RefDIOR/phrase_txts/val.json'))
# preds = json.load(open('qwen2vl_72b_refdiortest.json'))
#
# for pred in tqdm(preds):
#     if isinstance(pred, dict):
#         for k, v in pred.items():
#             numbers = re.findall(r'\d+\.\d+|\d+', v[0])
#             if numbers and len(numbers) ==4:
#                 pred_bbox = [float(i) for i in numbers]
#                 if sum(pred_bbox) > 4:
#                     pred_bbox = [coord / 999 for coord in pred_bbox]
#                 pred_bbox = np.array([coord * 800 for coord in pred_bbox])
#                 gt_bbox = np.array(gts[k]['bbox'])
#
#                 iou, i_area, u_area = bbox_iou(pred_bbox, gt_bbox)
#
#                 iou_5 += int(iou > .5)
#                 iou_6 += int(iou > .6)
#                 iou_7 += int(iou > .7)
#                 iou_8 += int(iou > .8)
#                 iou_9 += int(iou > .9)
#                 ious.append(iou)
#                 inter += i_area
#                 union += u_area
#
#     elif isinstance(pred, list):
#         idx = os.path.basename(pred[0]).split('.')[0]
#         gt_bbox = np.array(gts[idx]['bbox'])
#
#         numbers = re.findall(r'\d+\.\d+|\d+', pred[1])
#         if numbers and len(numbers) == 4:
#             pred_bbox = [float(i) for i in numbers]
#             if sum(pred_bbox) > 4:
#                 pred_bbox = [coord / 999 for coord in pred_bbox]
#             pred_bbox = np.array([coord * 800 for coord in pred_bbox])
#         else:
#             pred_bbox = [0., 0., 0., 0.]
#
#         iou, i_area, u_area = bbox_iou(pred_bbox, gt_bbox)
#
#         iou_5 += int(iou > .5)
#         iou_6 += int(iou > .6)
#         iou_7 += int(iou > .7)
#         iou_8 += int(iou > .8)
#         iou_9 += int(iou > .9)
#         ious.append(iou)
#         inter += i_area
#         union += u_area
#
#     else:
#         raise NotImplementedError
#
# print(len(ious))
# iou_5 = round(100 * iou_5 / len(preds), 2)
# iou_6 = round(100 * iou_6 / len(preds), 2)
# iou_7 = round(100 * iou_7 / len(preds), 2)
# iou_8 = round(100 * iou_8 / len(preds), 2)
# iou_9 = round(100 * iou_9 / len(preds), 2)
# miou = np.round(100 * np.mean(np.array(ious)), 2)
# ciou = round(100 * float(inter / union), 2)
#
# print(metric_table([iou_5, iou_6, iou_7, iou_8, iou_9, ciou, miou],
#                    ['PR@.5', 'PR@.6', 'PR@.7', 'PR@.8', 'PR@.9', 'cIoU', 'mIoU']))





# pred_path = 'qwen2vl_72b_refdiorval.json'
# preds = json.load(open(pred_path))
# convert_jsons = {}
# for pred in tqdm(preds):
#     if isinstance(pred, dict):
#         for k, v in pred.items():
#             numbers = re.findall(r'\d+\.\d+|\d+', v[0])
#             if numbers and len(numbers) == 4:
#                 pred_bbox = [float(i) for i in numbers]
#                 if sum(pred_bbox) > 4:
#                     pred_bbox = [coord / 999 for coord in pred_bbox]
#                 pred_bbox = [round(coord * 800, 2) for coord in pred_bbox]
#             else:
#                 pred_bbox = [0., 0., 0., 0.]
#             convert_jsons[k] = pred_bbox
#
#     elif isinstance(pred, list):
#         idx = os.path.basename(pred[0]).split('.')[0]
#
#         numbers = re.findall(r'\d+\.\d+|\d+', pred[1])
#         if numbers and len(numbers) == 4:
#             pred_bbox = [float(i) for i in numbers]
#             if sum(pred_bbox) > 4:
#                 pred_bbox = [coord / 999 for coord in pred_bbox]
#             pred_bbox = [round(coord * 800, 2) for coord in pred_bbox]
#         else:
#             pred_bbox = [0., 0., 0., 0.]
#
#         convert_jsons[idx] = pred_bbox
#
#     else:
#         raise NotImplementedError
#
# json.dump(convert_jsons, open('/root/lxq/RefDIOR_results/VLM/val/jsons/' + pred_path, 'w'))




# iou_5, iou_6, iou_7, iou_8, iou_9, ious, inter, union = 0., 0., 0., 0., 0., [], 0., 0.
#
# gts = json.load(open('/root/lxq/RSFM_RIS_Datasets/Optical/RefDIOR/phrase_txts/test.json'))
# preds = json.load(open('/root/data4/InternVL/internvl_chat/results/refdior_test_250104115237.json'))
#
# for pred in tqdm(preds):
#     numbers = re.findall(r'\d+\.\d+|\d+', pred['answer'])
#     if numbers and len(numbers) == 4:
#         pred_bbox = [float(i) for i in numbers]
#         pred_bbox = np.array([coord * 0.8 for coord in pred_bbox])
#         gt_bbox = np.array(pred['gt_bbox'])
#
#         iou, i_area, u_area = bbox_iou(pred_bbox, gt_bbox)
#
#         iou_5 += int(iou > .5)
#         iou_6 += int(iou > .6)
#         iou_7 += int(iou > .7)
#         iou_8 += int(iou > .8)
#         iou_9 += int(iou > .9)
#         ious.append(iou)
#         inter += i_area
#         union += u_area
#     else:
#         print(pred)
#
# print(len(ious))
# iou_5 = round(100 * iou_5 / len(preds), 2)
# iou_6 = round(100 * iou_6 / len(preds), 2)
# iou_7 = round(100 * iou_7 / len(preds), 2)
# iou_8 = round(100 * iou_8 / len(preds), 2)
# iou_9 = round(100 * iou_9 / len(preds), 2)
# miou = np.round(100 * np.mean(np.array(ious)), 2)
# ciou = round(100 * float(inter / union), 2)
#
# print(metric_table([iou_5, iou_6, iou_7, iou_8, iou_9, ciou, miou],
#                    ['PR@.5', 'PR@.6', 'PR@.7', 'PR@.8', 'PR@.9', 'cIoU', 'mIoU']))




gts = json.load(open('/root/lxq/RSFM_RIS_Datasets/Optical/RefDIOR/phrase_txts/val.json'))
preds = json.load(open('/root/data4/InternVL/internvl_chat/results/refdior_val_250104113713.json'))
cs = {}
for pred, gt in tqdm(zip(preds, gts)):

    numbers = re.findall(r'\d+\.\d+|\d+', pred['answer'])
    if numbers and len(numbers) == 4:
        pred_bbox = [round(float(i) * 0.8, 3) for i in numbers]
    else:
        pred_bbox = [0., 0., 0., 0.]

    cs[gt] = pred_bbox

json.dump(cs, open('/root/lxq/RefDIOR_results/InternVL2.5_2B_refdior_val.json', 'w'))
