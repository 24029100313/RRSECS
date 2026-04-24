import json
import torch
from prettytable import PrettyTable
from tqdm import tqdm
import numpy as np


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = torch.tensor(box1[0]), torch.tensor(box1[1]), torch.tensor(box1[2]), torch.tensor(box1[3])
    b2_x1, b2_y1, b2_x2, b2_y2 = torch.tensor(box2[0]), torch.tensor(box2[1]), torch.tensor(box2[2]), torch.tensor(box2[3])

    # get the coordinates of the intersection rectangle

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    return inter_area / (union_area + 1e-10), inter_area, union_area


def metric_table(metrics, metrics_name, need_perclass=False, classes_name=None):
    table = PrettyTable()
    table.field_names = metrics_name

    if need_perclass:
        for i, class_name in enumerate(classes_name):
            table.add_row([class_name, round(metrics[0][i], 2), round(metrics[1][i], 2), '-'])
        table.add_row(['Average', round(metrics[2], 2), round(metrics[3], 2), round(metrics[4], 2)])
    else:
        table.add_row([round(metric, 2) for metric in metrics])

    return table


def vg_off_eval(pred_file, gt_file, verbose=True):
    annos = json.load(open(gt_file))
    # ids = annos.keys()
    preds = json.load(open(pred_file))
    ids = preds.keys()

    iou_5, iou_6, iou_7, iou_8, iou_9, ious, inter, union = 0., 0., 0., 0., 0., [], 0., 0.,
    for id in tqdm(ids):
        gt_bbox, pred_bbox = np.array(annos[id]['bbox']), np.array(preds[id])

        # gt_bbox = np.array(annos[id]['bbox'])
        # if id in preds.keys():
        #     pred_bbox = np.array(preds[id])
        # else:
        #     pred_bbox = np.array([0., 0., 0., 0.])

        iou, i_area, u_area = bbox_iou(pred_bbox, gt_bbox)

        iou_5 += int(iou > .5)
        iou_6 += int(iou > .6)
        iou_7 += int(iou > .7)
        iou_8 += int(iou > .8)
        iou_9 += int(iou > .9)
        ious.append(iou)
        inter += i_area
        union += u_area

    iou_5 = round(100 * iou_5 / len(ids), 2)
    iou_6 = round(100 * iou_6 / len(ids), 2)
    iou_7 = round(100 * iou_7 / len(ids), 2)
    iou_8 = round(100 * iou_8 / len(ids), 2)
    iou_9 = round(100 * iou_9 / len(ids), 2)
    miou = np.round(100 * np.mean(np.array(ious)), 2)
    ciou = round(100 * float(inter / union), 2)
    total = round(sum([iou_5, iou_6, iou_7, iou_8, iou_9, ciou, miou]), 2)

    if verbose:
        print(metric_table([iou_5, iou_6, iou_7, iou_8, iou_9, ciou, miou, total],
                           ['PR@.5', 'PR@.6', 'PR@.7', 'PR@.8', 'PR@.9', 'cIoU', 'mIoU', 'Sum']))
    else:
        return [iou_5, iou_6, iou_7, iou_8, iou_9, ciou, miou, total]


def main():
    pred_json = input('please enter your prediction dir: ')
    gt_json = input('please enter your groundtruth selection from 1: DIOR-RSVG, 2: OPT-RSVG, 3: RefDIOR, 4: RefCOCO\nYour choice: ')

    if '1' == gt_json:
        gt_path = '/root/lxq/RSFM_VG_Datasets/Optical/DIOR-RSVG/phrase_txts/test_bbox.json'
    elif '2' == gt_json:
        gt_path = '/root/lxq/RSFM_VG_Datasets/Optical/OPT-RSVG/phrase_txts/test_bbox.json'
    elif '3' == gt_json:
        gt_path = '/root/lxq/RSFM_RIS_Datasets/Optical/RefDIOR/phrase_txts/val.json'
    elif '4' == gt_json:
        gt_path = '/root/lxq/RSFM_RIS_Datasets/Natural/RefCOCO/phrase_txts/val.json'
    else:
        raise NotImplementedError

    vg_off_eval(gt_path, pred_json)

if __name__ == '__main__':
    main()