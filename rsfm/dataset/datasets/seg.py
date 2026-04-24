import math
from torch.utils.data import Dataset
from rsfm.dataset.utils import *
from rsfm.dataset.transform.transform import *
from os.path import join as osp



class create_seg_dataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.root, self.size, self.img_suffix, self.mask_suffix, \
        self.num_bands, self.need_bands, self.reduce_zero_label = base_init(cfg)

        self.mode = mode

        self.semi_supervised = cfg.get('semi_supervised', None)
        if self.semi_supervised:
            labeled_repeat_samples_dict = {'PASCAL_VOC': 3000, 'Cityscapes': 3000, 'ADE20K': 6000, 'COCO': 30000}
            splits_path = 'rsfm/dataset/splits/' + cfg['dataset']
            if mode == 'train':
                self.ids = [img_id.strip() for img_id in open(
                    osp(splits_path, str(self.semi_supervised.get('labeled_partition')) + '_labeled.txt'), 'r').readlines()]
                nsample = labeled_repeat_samples_dict[cfg['dataset']]
                if nsample > len(self.ids):
                    self.ids *= math.ceil(nsample / len(self.ids))
                    self.ids = self.ids[:nsample]
            else:
                self.ids = [img_id.strip() for img_id in open(
                    osp(splits_path, 'val.txt'), 'r').readlines()]
        else:
            self.ids = [img_id.split(self.img_suffix)[0] for img_id in os.listdir(osp(self.root, self.mode, 'images'))]

    def __getitem__(self, item):
        id = self.ids[item]

        img, mask = load_img_mask_file(
            need_bands=self.need_bands,
            img_dir=osp(self.root, '' if self.semi_supervised else self.mode, 'images/' + id + self.img_suffix),
            mask_dir=osp(self.root, '' if self.semi_supervised else self.mode, 'labels/' + id + self.mask_suffix),
            reduce_zero_label=self.reduce_zero_label)

        if self.mode == 'train':
            img, mask = resize(img, mask, self.size, (0.5, 2.0))
            img, mask = crop(img, mask, self.size, 255)
            img, mask = hflip(img, mask, p=0.5)
        elif self.mode == 'val':
            img, mask = resize_fixed(img, mask, self.size)
            # img, mask = resize_and_pad(img, mask, self.size)
        else:
            raise NotImplementedError('%s mode is not implemented' % self.mode)

        img, mask = normalize(img, mask, self.num_bands)

        return img, mask

    def __len__(self):
        return len(self.ids)
