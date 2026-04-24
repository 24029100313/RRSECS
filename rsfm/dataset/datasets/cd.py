import os
from torch.utils.data import Dataset
from rsfm.dataset.utils import *
from rsfm.dataset.transform.transform import *
from os.path import join as osp



class create_cd_dataset(Dataset):
    def __init__(self, cfg, mode='train', txt_path=False):
        self.root, self.size, self.img_suffix, self.mask_suffix, \
        self.num_bands, self.need_bands, self.reduce_zero_label = base_init(cfg)

        self.mode = mode

        if not txt_path:
            self.ids = [img_id.split(self.img_suffix)[0] for img_id in os.listdir(osp(self.root, mode, 'images1'))]
        else:
            self.ids = [img_id.strip().split(self.img_suffix)[0] for img_id in open(txt_path, 'r').readlines()]

    def __getitem__(self, item):
        id = self.ids[item]

        img1, img2, mask = load_img1_img2_mask_file(
            need_bands=self.need_bands,
            img1_dir=osp(self.root, self.mode, 'images1/' + id + self.img_suffix),
            img2_dir=osp(self.root, self.mode, 'images2/' + id + self.img_suffix),
            mask_dir=osp(self.root, self.mode, 'labels/' + id + self.mask_suffix),
            reduce_zero_label=self.reduce_zero_label)

        if self.mode == 'train':
            img1, img2, mask = resize2(img1, img2, mask, self.size, (0.5, 2.0))
            img1, img2, mask = crop2(img1, img2, mask, self.size, 255)
            img1, img2, mask = hflip2(img1, img2, mask, p=0.5)
        elif self.mode == 'val':
            img1, img2, mask = resize2_fixed(img1, img2, mask, self.size)
        else:
            raise NotImplementedError('%s mode is not implemented' % self.mode)

        img1, img2, mask = normalize2(img1, img2, mask, self.num_bands)

        return img1, img2, mask

    def __len__(self):
        return len(self.ids)