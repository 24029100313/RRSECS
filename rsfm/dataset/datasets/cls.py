from torch.utils.data import Dataset
from rsfm.dataset.utils import *
from rsfm.dataset.transform.transform import *
from os.path import join as osp
import glob



class create_cls_dataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.root, self.size, self.img_suffix, self.mask_suffix, \
        self.num_bands, self.need_bands, self.reduce_zero_label = base_init(cfg)

        self.mode = mode

        self.classes_dict = {cat: i for i, cat in enumerate(datasets_info[cfg['dataset']]['classes_name'])}
        self.ids = glob.glob(osp(self.root, mode, '**/*' + self.img_suffix), recursive=True)

        self.use_strong_aug = cfg['strong_aug']
        if self.use_strong_aug:
            self.strong_aug = build_transform(cfg)

    def __getitem__(self, item):
        id = self.ids[item]

        img = load_img_mask_file(need_bands=self.need_bands, img_dir=id,
                                 mask_dir=None, need_mask=False)
        label = self.classes_dict[id.split('/')[-2]]

        if self.mode == 'train':
            img = resize(img, None, self.size, (0.5, 2.0), interpolation='bicubic')
            img = crop(img, None, self.size, 255)
            img = hflip(img, p=0.5)
            if self.use_strong_aug:
                img = self.strong_aug(img)
                return img, label
        elif self.mode == 'val':
            # img = resize_fixed(img, None, self.size)
            img = resize_pct_center(img, self.size, interpolation='bicubic')
        else:
            raise NotImplementedError('%s mode is not implemented' % self.mode)

        img = normalize(img, num_bands=self.num_bands)

        return img, label

    def __len__(self):
        return len(self.ids)