import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import cv2
import torch
import numpy as np
from prettytable import PrettyTable


def IoU(pred, gt):
    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union


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


def ris_off_eval(pred_path, gt_path, gt_suffix='.png', verbose=True):
    def process_file(file):
        nonlocal sum_I, sum_U, seg_total, seg_correct, miou

        pred = cv2.imread(pred_path + '/' + file, 0)
        if 255 in pred:
            pred = (pred / 255).astype(np.uint8)
        gt = cv2.imread(gt_path + '/' + file.split('.')[0] + gt_suffix, 0)
        if 255 in gt:
            gt = (gt / 255).astype(np.uint8)

        if pred.shape != gt.shape:
            gt = cv2.resize(gt, pred.shape[::-1], interpolation=cv2.INTER_NEAREST)

        pred, gt = torch.from_numpy(pred).unsqueeze(0), torch.from_numpy(gt)

        iou, I, U = IoU(pred, gt)

        miou.append(iou)
        sum_I += I
        sum_U += U

        for n_eval_iou in range(len(eval_seg_iou_list)):
            eval_seg_iou = eval_seg_iou_list[n_eval_iou]
            seg_correct[n_eval_iou] += (iou >= eval_seg_iou)

        seg_total += 1

    metrics, miou, sum_I, sum_U, seg_total = [], [], 0, 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)

    files = os.listdir(pred_path)
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_file, files), total=len(files)))

    mean_IoU = np.array(miou)
    mIoU = np.mean(mean_IoU)

    for n_eval_iou in range(len(eval_seg_iou_list)):
        metrics.append(seg_correct[n_eval_iou] * 100. / seg_total)
    metrics.append(float(sum_I * 100. / sum_U))
    metrics.append(mIoU * 100)
    sum_m = sum(metrics)
    metrics.append(sum_m)

    if verbose:
        print(metric_table(metrics, ['PR@.5', 'PR@.6', 'PR@.7', 'PR@.8', 'PR@.9', 'oIoU', 'mIoU', 'Sum']))
    else:
        return metrics



def main():
    pred_path = input('please enter your prediction dir: ')
    label_path = input('please enter your groundtruth selection from 1: RefSegRS, 2: RRSIS-D, 3: RefDIOR\nYour choice: ')

    if '1' == label_path:
        gt_path = '/root/lxq/RSFM_RIS_Datasets/Optical/RefSegRS/masks'
        gt_suffix = '.tif'
    elif '2' == label_path:
        gt_path = '/root/lxq/RSFM_RIS_Datasets/Optical/RRSIS-D/masks'
        gt_suffix = '.png'
    elif '3' == label_path:
        gt_path = '/root/lxq/RSFM_RIS_Datasets/Optical/RefDIOR/masks'
        gt_suffix = '.png'
    else:
        raise NotImplementedError

    ris_off_eval(pred_path, gt_path, gt_suffix)


if __name__ == '__main__':
    main()