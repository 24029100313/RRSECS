from rsfm.dataset.transform.transform import *
from rsfm.dataset.utils import CocoDetection
from os.path import join as osp


def create_od_dataset(cfg, mode='train'):
    dataset_info = datasets_info.get(cfg['dataset'])
    root = cfg.get('data_root', dataset_info.get('data_root'))

    img_folder = osp(root, 'images')
    if mode == 'train':
        ann_file = osp(root, 'annotations', 'train.json')
    elif mode == 'val':
        ann_file = osp(root, 'annotations', 'test.json')
    else:
        raise NotImplementedError

    transform = make_od_transforms(cfg['data_aug'], mode)

    dataset = CocoDetection(img_folder=img_folder,
                            ann_file=ann_file,
                            transforms=transform)

    return dataset