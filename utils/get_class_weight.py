import math

import numpy as np
import cv2, os
from tqdm import tqdm
from PIL import Image


def get_class_weight_median(path, num_classes, txt_path=None, reduce_zero=False):
    pixel_sum = np.zeros((num_classes), dtype=np.int64)
    if txt_path == None:
        for file in tqdm(os.listdir(path)):
            img = np.array(Image.open(path + '/' + file))
            if reduce_zero:
                img -= 1
            for i in range(num_classes):
                pixel_sum[i] += np.sum(img == i)
    else:
        names = open(txt_path, 'r').readlines()
        for name in tqdm(names):
            img = np.array(Image.open(path + '/' + name.strip()))
            if reduce_zero:
                img -= 1
            for i in range(num_classes):
                pixel_sum[i] += np.sum(img == i)

    class_freq = np.divide(pixel_sum, np.sum(pixel_sum))
    print(class_freq)
    median = np.median(class_freq)
    class_weight = np.array(median / class_freq)

    return class_weight


def get_class_weight_log(path, num_classes, reduce_zero=False):
    z = np.zeros((num_classes,))

    for file in tqdm(os.listdir(path)):
        img = np.array(Image.open(path + '/' + file))
        if reduce_zero:
            img -= 1
        mask = (img >= 0) & (img < num_classes)
        labels = img[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l

    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)

    return ret





class_weight = get_class_weight_median(path='/root/lxq/RSFM_MMSeg_Datasets/MMSeg-YREB/train/labels',
                                       num_classes=9, reduce_zero=True,
                                       # txt_path='/root/lxq/Agriculture_Vision/train/train_aug.txt'
                                       )
print(class_weight)


