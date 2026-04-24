import glob
import math
import os
import os.path as osp
import mmcv
import numpy as np

input_dir = '/root/Download/Potsdam/DSM'
save_dir = '/root/Download/Potsdam/DSM_silde'
window_size = 512
stride_size = 256
threads = 32  # multithreading

png_files = glob.glob(os.path.join(input_dir, '**/*.jpg'), recursive=True)


def clip_big_image(image_path):
    image = mmcv.imread(image_path)
    clip_save_dir = os.path.join(save_dir, os.path.relpath(os.path.dirname(image_path), input_dir))
    os.makedirs(os.path.dirname(clip_save_dir), exist_ok=True)
    to_label = False
    if 'label' in image_path:
        to_label = True

    h, w, c = image.shape

    num_rows = math.ceil((h - window_size) / stride_size) if math.ceil(
        (h - window_size) /
        stride_size) * stride_size + window_size >= h else math.ceil(
        (h - window_size) / stride_size) + 1
    num_cols = math.ceil((w - window_size) / stride_size) if math.ceil(
        (w - window_size) /
        stride_size) * stride_size + window_size >= w else math.ceil(
        (w - window_size) / stride_size) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * window_size
    ymin = y * window_size

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + window_size > w, w - xmin - window_size,
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + window_size > h, h - ymin - window_size,
                           np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + window_size, w),
        np.minimum(ymin + window_size, h)
    ], axis=1)

    if to_label:
        image[image == 255] = 1
        image = image[:, :, 0]
    for box in boxes:
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y, start_x:end_x] \
            if to_label else image[start_y:end_y, start_x:end_x, :]
        idx = osp.basename(image_path).split('.')[0]
        mmcv.imwrite(
            clipped_image.astype(np.uint8),
            osp.join(clip_save_dir,
                     f'{idx}_{start_x}_{start_y}_{end_x}_{end_y}.png'))


mmcv.track_parallel_progress(clip_big_image, png_files, threads)
