import glob

import numpy as np
import torch

from rsfm.dataset.transform.transform import *
from rsfm.dataset.utils import *
from os.path import join as osp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from rsfm.language import BertTokenizer as RISBertTokenizer
from pytorch_pretrained_bert.tokenization import BertTokenizer
from .datasets_info import datasets_info
from .datasets import create_seg_dataset, create_cd_dataset, create_cls_dataset, \
    create_ris_dataset, create_vg_dataset, create_od_dataset, create_recs_dataset



def create_dataset(cfg, rank, logger):
    task = datasets_info[cfg['dataset']]['vision_task']

    if task in ['semantic segmentation', 'seg']:
        trainset, valset = create_seg_dataset(cfg, mode='train'), create_seg_dataset(cfg, mode='val')

    elif task in ['change detection', 'cd']:
        trainset, valset = create_cd_dataset(cfg, mode='train'), create_cd_dataset(cfg, mode='val')

    elif task in ['scene classification', 'cls']:
        trainset, valset = create_cls_dataset(cfg, mode='train'), create_cls_dataset(cfg, mode='val')

    elif task in ['referring image segmentation', 'ris']:
        trainset, valset = create_ris_dataset(cfg, mode='train'), create_ris_dataset(cfg, mode='val')

    elif task in ['visual grounding', 'vg']:
        trainset, valset = create_vg_dataset(cfg, mode='train'), create_vg_dataset(cfg, mode='val')

    elif task in ['object detection', 'od']:
        trainset, valset = create_od_dataset(cfg, mode='train'), create_od_dataset(cfg, mode='val')

    elif task in ['referring expression comprehension and segmentation', 'recs']:
        trainset, valset = create_recs_dataset(cfg, mode='train'), create_recs_dataset(cfg, mode='val')

    else:
        raise NotImplementedError

    if rank == 0:
        logger.info('Finding {:} images for training'.format(len(trainset)))
        logger.info('Finding {:} images for validation'.format(len(valset)))

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)

    if task == 'object detection':
        batch_trainsampler = torch.utils.data.BatchSampler(trainsampler, cfg['batch_size'], drop_last=True)
        trainloader = DataLoader(trainset, batch_sampler=batch_trainsampler,
                                 collate_fn=collate_fn, num_workers=8)
        valloader = DataLoader(valset, batch_size=1, num_workers=8, drop_last=False,
                               sampler=valsampler, collate_fn=collate_fn)
    else:
        trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], pin_memory=False,
                                 num_workers=8, drop_last=True, sampler=trainsampler)
        valloader = DataLoader(valset, batch_size=cfg['batch_size'] if task == 'scene classification' else 1,
                               pin_memory=False, num_workers=8, drop_last=False, sampler=valsampler)

    return trainloader, valloader, trainsampler, valset


class create_test_dataset(Dataset):
    def __init__(self, task, dataset, mode='val', infer_size=None, txt_path=False):
        self.task = task
        self.dataset = dataset
        self.mode = mode
        txt_path = txt_path
        self.size = infer_size

        dataset_info = datasets_info[self.dataset]

        self.root = dataset_info.get('data_root')
        self.need_bands = dataset_info.get('need_bands')
        self.img_suffix = dataset_info.get('img_suffix')
        self.mask_suffix = dataset_info.get('mask_suffix')
        self.vision_task = dataset_info.get('vision_task')
        self.reduce_zero_label = dataset_info.get('reduce_zero_label')
        self.convert_255_1 = dataset_info.get('convert_255_1', False)

        self.num_bands = dataset_info.get('num_bands')
        if self.num_bands <= 3:
            self.num_bands = 3
        elif self.num_bands > 4:
            self.num_bands = len(self.need_bands)
        else:
            self.num_bands = 4

        self.classes_dict = {cat: i for i, cat in enumerate(dataset_info.get('classes_name'))}

        if self.vision_task in ['referring image segmentation', 'ris',
                                'visual grounding', 'vg',
                                'referring expression comprehension and segmentation', 'recs']:
            txt_path = osp(self.root, 'phrase_txts',
                           'test_mbr.txt' if self.dataset not in ['RefDIOR_RIS', 'RefCOCO', 'RefCOCO+', 'G-Ref-g', 'G-Ref-u',
                                                              'RefDIOR_VG', 'RefCOCO_VG', 'RefCOCO+_VG', 'G-Ref-g_VG', 'G-Ref-u_VG',
                                                              'RefDIOR_RECS']
                           else 'val_ablation.json')

        if not txt_path:
            if self.vision_task in ['semantic segmentation', 'seg']:
                self.ids = [img_id.split(self.img_suffix)[0] for img_id in
                            os.listdir(osp(self.root, self.mode, 'images'))]

            elif self.vision_task in ['change detection', 'cd']:
                self.ids = [img_id.split(self.img_suffix)[0] for img_id in
                            os.listdir(osp(self.root, self.mode, 'images1'))]

            elif self.vision_task in ['scene classification', 'cls']:
                self.ids = glob.glob(osp(self.root, self.mode, '**/*' + self.img_suffix),
                                     recursive=True)

            else:
                raise NotImplementedError
        else:
            if self.vision_task in ['referring image segmentation', 'ris']:
                self.max_tokens = 20
                self.tokenizer = RISBertTokenizer.from_pretrained('pretrained/bert/bert-base-uncased',
                                                                  do_lower_case=True)
                self.uni_ids, self.ids, self.mask_ids, self.phrases = \
                    load_id_phrase_mask_info(self.root, txt_path, self.img_suffix, self.mask_suffix)
                self.transform = make_vlm_transforms(self.size, self.mode)

            elif self.vision_task == 'visual grounding':
                self.max_tokens = 20
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
                self.uni_ids, self.ids, self.bboxs, self.phrases = load_id_phrase_bbox_info(txt_path, self.img_suffix)
                self.transform = make_vlm_transforms(self.size, self.mode)

            elif self.vision_task in ['referring expression comprehension and segmentation', 'recs']:
                self.max_tokens = 20
                self.tokenizer = RISBertTokenizer.from_pretrained('pretrained/bert/bert-base-uncased',
                                                                  do_lower_case=True)
                self.uni_ids, self.ids, self.mask_ids, self.bboxs, self.phrases = \
                    load_id_phrase_mask_bbox_info(self.root, txt_path, self.img_suffix, self.mask_suffix)
                self.transform = make_vlm_transforms(self.size, self.mode)

            else:
                self.ids = [img_id.strip().split(self.img_suffix)[0] for img_id in open(txt_path, 'r').readlines()]

    def __getitem__(self, item):
        id = self.ids[item]

        if self.vision_task in ['semantic segmentation', 'seg']:
            if self.task == 'eval':
                img, mask = load_img_mask_file(need_bands=self.need_bands,
                                               img_dir=osp(self.root, self.mode, 'images/' + id + self.img_suffix),
                                               mask_dir=osp(self.root, self.mode, 'labels/' + id + self.mask_suffix),
                                               reduce_zero_label=self.reduce_zero_label)
                img, mask = resize_fixed(img, mask, self.size)
                img, mask = normalize(img, mask, self.num_bands)

                return img, mask
            else:
                img = load_img_mask_file(need_bands=self.need_bands,
                                         img_dir=osp(self.root, self.mode, 'images/' + id + self.img_suffix),
                                         mask_dir=None, need_mask=False)
                img = resize_fixed(img, mask=None, size=self.size)

                return normalize(img, num_bands=self.num_bands), id

        elif self.vision_task in ['change detection', 'cd']:
            if self.task == 'eval':
                img1, img2, mask = load_img1_img2_mask_file(
                    need_bands=self.need_bands,
                    img1_dir=osp(self.root, self.mode, 'images1/' + id + self.img_suffix),
                    img2_dir=osp(self.root, self.mode, 'images2/' + id + self.img_suffix),
                    mask_dir=osp(self.root, self.mode, 'labels/' + id + self.mask_suffix),
                    reduce_zero_label=self.reduce_zero_label)
                img1, img2, mask = resize2_fixed(img1, img2, mask, self.size)
                img1, img2, mask = normalize2(img1, img2, mask, self.num_bands)

                return img1, img2, mask
            else:
                img1, img2 = load_img1_img2_mask_file(
                    need_bands=self.need_bands,
                    img1_dir=osp(self.root, self.mode, 'images1/' + id + self.img_suffix),
                    img2_dir=osp(self.root, self.mode, 'images2/' + id + self.img_suffix),
                    mask_dir=None, need_mask=False)
                img1, img2 = resize2_fixed(img1, img2, mask=None, size=self.size)

                return normalize2(img1, img2, num_bands=self.num_bands), id

        elif self.vision_task in ['scene classification', 'cls']:
            if self.task == 'eval':
                img = load_img_mask_file(need_bands=self.need_bands, img_dir=id,
                                         mask_dir=None, need_mask=False)
                label = self.classes_dict[id.split('/')[-2]]

                # img = resize_fixed(img, None, self.size)
                img = resize_pct_center(img, self.size, interpolation='bicubic')
                img = normalize(img, num_bands=self.num_bands)

                return img, label
            else:
                img = load_img_mask_file(need_bands=self.need_bands, img_dir=id,
                                         mask_dir=None, need_mask=False) # may cause bug
                # img = resize_fixed(img, None, self.size)
                img = resize_pct_center(img, self.size, interpolation='bicubic')

                return normalize(img, num_bands=self.num_bands), id

        elif self.vision_task in ['referring image segmentation', 'ris']:
            img, mask = load_img_mask_file(need_bands=self.need_bands,
                                           img_dir=id,
                                           mask_dir=self.mask_ids[item],
                                           convert_255_1=True)
            phrase = self.phrases[item].lower()

            target = {}
            target['phrase'] = phrase
            target['ris_mask'] = mask

            img, target = self.transform(img, target)

            mask = target.pop('ris_mask')

            attention_mask = [0] * self.max_tokens
            padded_input_id = [0] * self.max_tokens

            input_id = self.tokenizer.encode(text=target['phrase'], add_special_tokens=True)
            input_id = input_id[:self.max_tokens]

            padded_input_id[:len(input_id)] = input_id
            attention_mask[:len(input_id)] = [1] * len(input_id)

            tensor_embeddings = torch.tensor(padded_input_id).unsqueeze(0)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0)

            if self.task == 'eval':
                return img, mask, tensor_embeddings, attention_mask
            else:
                return img, tensor_embeddings, attention_mask, os.path.basename(id).split('.')[0]

        elif self.vision_task == 'visual grounding':
            img = load_img_mask_file(need_bands=self.need_bands,
                                     img_dir=id, mask_dir=None, need_mask=False)

            bbox = torch.tensor(self.bboxs[item]).float()
            phrase = self.phrases[item].lower()

            target = {}
            target['phrase'] = phrase
            target['bbox'] = bbox

            orig_bbox = bbox.clone()

            img, target = self.transform(img, target)

            examples = read_examples(target['phrase'], item)
            features = convert_examples_to_features(
                examples=examples, seq_length=self.max_tokens, tokenizer=self.tokenizer)
            word_id = torch.tensor(features[0].input_ids, dtype=torch.long)
            word_mask = torch.tensor(features[0].input_mask, dtype=torch.bool)

            mask = target.pop('mask')
            size, ratio, orig_size, dxdy = target.pop('size'), target.pop('ratio'), \
                                           target.pop('orig_size'), target.pop('dxdy')

            return img, mask, word_id, word_mask, orig_bbox, size, ratio, orig_size, dxdy, os.path.basename(id).split('.')[0]

        elif self.vision_task in ['referring expression comprehension and segmentation', 'recs']:
            img, mask = load_img_mask_file(need_bands=self.need_bands,
                                           img_dir=id,
                                           mask_dir=self.mask_ids[item],
                                           convert_255_1=True)

            # img = load_img_mask_file(need_bands=self.need_bands, img_dir=id,
            #                          mask_dir=self.mask_ids[item], convert_255_1=True, need_mask=False)
            # w, h = img.size
            # rec_mask = np.zeros((h, w), dtype=np.uint8)
            # x1, y1, x2, y2 = self.bboxs[item]
            # rec_mask[y1:y2, x1:x2] = 1
            # mask = Image.fromarray(rec_mask)

            bbox = torch.tensor(self.bboxs[item]).float()
            phrase = self.phrases[item].lower()

            target = {}
            target['phrase'] = phrase
            target['bbox'] = bbox
            target['ris_mask'] = mask

            orig_bbox = bbox.clone()
            orig_mask = torch.from_numpy(np.array(mask)).long()

            img, target = self.transform(img, target)

            attention_mask = [0] * self.max_tokens
            padded_input_id = [0] * self.max_tokens

            input_id = self.tokenizer.encode(text=target['phrase'], add_special_tokens=True)
            input_id = input_id[:self.max_tokens]

            padded_input_id[:len(input_id)] = input_id
            attention_mask[:len(input_id)] = [1] * len(input_id)

            word_id = torch.tensor(padded_input_id).unsqueeze(0)
            word_mask = torch.tensor(attention_mask).unsqueeze(0)

            img_mask = target.pop('mask')  # binary mask for augmented img areas
            size = target.pop('size')
            ratio = target.pop('ratio')
            orig_size = target.pop('orig_size')
            dxdy = target.pop('dxdy')

            return img, img_mask, word_id, word_mask, orig_mask, orig_bbox, \
                   size, ratio, orig_size, dxdy, os.path.basename(id).split('.')[0]
            # return img, img_mask, word_id, word_mask, orig_mask, orig_bbox, \
            #        size, ratio, orig_size, dxdy, self.uni_ids[item]

        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.ids)