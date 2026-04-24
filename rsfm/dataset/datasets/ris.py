from torch.utils.data import Dataset
from rsfm.dataset.utils import *
from rsfm.dataset.transform.transform import *
from os.path import join as osp
from rsfm.language import BertTokenizer



class create_ris_dataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.root, self.size, self.img_suffix, self.mask_suffix, \
        self.num_bands, self.need_bands, self.reduce_zero_label, self.convert_255_1 = base_init(cfg)

        self.max_tokens = 20
        self.tokenizer = BertTokenizer.from_pretrained('pretrained/bert/' + cfg['model']['text_encoder']['type'])

        self.mode = mode
        if mode == 'train':
            txt_path = osp(self.root, 'phrase_txts',
                           'trainval.txt' if cfg['dataset'] not in ['RefDIOR_RIS', 'RefCOCO', 'RefCOCO+', 'G-Ref-g', 'G-Ref-u']
                           else 'abla_splits/1/labeled.json')
        elif mode == 'val':
            txt_path = osp(self.root, 'phrase_txts',
                           'test.txt' if cfg['dataset'] not in ['RefDIOR_RIS', 'RefCOCO', 'RefCOCO+', 'G-Ref-g', 'G-Ref-u']
                           else 'abla_splits/val_ablation.json')
        else:
            raise NotImplementedError

        self.uni_ids, self.img_ids, self.mask_ids, self.phrases = \
            load_id_phrase_mask_info(self.root, txt_path, self.img_suffix, self.mask_suffix)

        # self.transform = make_vlm_transforms(self.size, self.mode, cfg.get('strong_aug', False))

        scales = [self.size - (32 * i) for i in range(6, -1, -1)]
        self.transform = TVLM.Compose([TVLM.RandomResize(scales),
                                       TVLM.RandomHorizontalFlip(p=0.5),
                                       TVLM.ToTensor(),
                                       TVLM.NormalizeAndPad(size=self.size, aug_translate=True)])

    def __getitem__(self, item):
        img, mask = load_img_mask_file(need_bands=self.need_bands,
                                       img_dir=self.img_ids[item],
                                       mask_dir=self.mask_ids[item],
                                       convert_255_1=True)
        phrase = self.phrases[item].lower()

        target = {}
        target['phrase'] = phrase
        target['ris_mask'] = mask

        # orig_mask = torch.from_numpy(np.array(mask)).long()

        img, target = self.transform(img, target)

        mask = target.pop('ris_mask')

        tensor_embeddings, attention_mask = self._tokenize(target['phrase'])

        return img, mask, tensor_embeddings, attention_mask

    def __len__(self):
        return len(self.uni_ids)

    def _tokenize(self, phrase):
        attention_mask = [0] * self.max_tokens
        padded_input_id = [0] * self.max_tokens

        input_id = self.tokenizer.encode(text=phrase, add_special_tokens=True)
        input_id = input_id[:self.max_tokens]

        padded_input_id[:len(input_id)] = input_id
        attention_mask[:len(input_id)] = [1] * len(input_id)

        return torch.tensor(padded_input_id).unsqueeze(0), torch.tensor(attention_mask).unsqueeze(0)



class create_semi_ris_dataset(Dataset):
    def __init__(self, cfg, mode='train_l', split='1', nsample=None):
        self.root, self.size, self.img_suffix, self.mask_suffix, \
        self.num_bands, self.need_bands, self.reduce_zero_label, self.convert_255_1 = base_init(cfg)

        self.max_tokens = 20
        self.tokenizer = BertTokenizer.from_pretrained('pretrained/bert/' + cfg['model']['text_encoder']['type'])

        self.mode = mode
        if mode == 'train_l':
            txt_path = osp(self.root, 'phrase_txts/abla_splits', split + '/labeled.json')
        elif mode == 'train_u':
            txt_path = osp(self.root, 'phrase_txts/abla_splits', split + '/unlabeled.json')
        elif mode == 'val':
            txt_path = osp(self.root, 'phrase_txts/abla_splits', 'val_ablation.json')
        else:
            raise NotImplementedError

        self.uni_ids, self.img_ids, self.mask_ids, self.phrases = \
            load_id_phrase_mask_info(self.root, txt_path, self.img_suffix, self.mask_suffix)

        if self.mode == 'train_l' and nsample is not None and nsample > len(self.uni_ids):
            resample_ratio = math.ceil(nsample / len(self.uni_ids))

            self.uni_ids *= resample_ratio
            self.uni_ids = self.uni_ids[:nsample]

            self.img_ids *= resample_ratio
            self.img_ids = self.img_ids[:nsample]

            self.mask_ids *= resample_ratio
            self.mask_ids = self.mask_ids[:nsample]

            self.phrases *= resample_ratio
            self.phrases = self.phrases[:nsample]

        self.transform_val = TVLM.Compose([TVLM.RandomResize([self.size], record_resize_info=True),
                                           TVLM.ToTensor(),
                                           TVLM.NormalizeAndPad(size=self.size, center_place=True)])

        scales = [self.size - (32 * i) for i in range(6, -1, -1)]
        self.transform_weakaug = TVLM.Compose([TVLM.RandomResize(scales),
                                               TVLM.RandomHorizontalFlip(p=0.5)])
        self.transform_strongaug = TVLM.Compose([TVLM.ColorJitter(p=0.8),
                                                 TVLM.EdgeEnhance(p=0.5),
                                                 TVLM.RandomGray(p=0.2),
                                                 TVLM.GaussianBlur(p=0.5)])
        self.transform_normalize = TVLM.Compose([TVLM.ToTensor(),
                                                 TVLM.NormalizeAndPad(size=self.size, aug_translate=True)])

    def __getitem__(self, item):
        img, mask = load_img_mask_file(need_bands=self.need_bands,
                                       img_dir=self.img_ids[item],
                                       mask_dir=self.mask_ids[item],
                                       convert_255_1=True)
        phrase = self.phrases[item].lower()

        target = {}
        target['phrase'] = phrase
        target['ris_mask'] = mask

        # orig_mask = torch.from_numpy(np.array(mask)).long()

        if self.mode == 'val':
            img, target = self.transform_val(img, target)
            mask = target.pop('ris_mask')
            tensor_embeddings, attention_mask = self._tokenize(target['phrase'])
            return img, mask, tensor_embeddings, attention_mask
        else:
            img, target = self.transform_weakaug(img, target)
            if self.mode == 'train_l':
                img, target = self.transform_normalize(img, target)
                mask = target.pop('ris_mask')
                tensor_embeddings, attention_mask = self._tokenize(target['phrase'])
                return img, mask, tensor_embeddings, attention_mask
            else:
                img_s = self.transform_strongaug(img)
                img_s = self.transform_normalize(img_s)
                img, target = self.transform_normalize(img, target)
                tensor_embeddings, attention_mask = self._tokenize(target['phrase'])
                return img, img_s, tensor_embeddings, attention_mask

                # img_s1 = self.transform_normalize(self.transform_strongaug(img))
                # img_s2 = self.transform_normalize(self.transform_strongaug(img))
                # img, target = self.transform_normalize(img, target)
                # tensor_embeddings, attention_mask = self._tokenize(target['phrase'])
                # return img, img_s1, img_s2, tensor_embeddings, attention_mask

    def __len__(self):
        return len(self.uni_ids)

    def _tokenize(self, phrase):
        attention_mask = [0] * self.max_tokens
        padded_input_id = [0] * self.max_tokens

        input_id = self.tokenizer.encode(text=phrase, add_special_tokens=True)
        input_id = input_id[:self.max_tokens]

        padded_input_id[:len(input_id)] = input_id
        attention_mask[:len(input_id)] = [1] * len(input_id)

        return torch.tensor(padded_input_id).unsqueeze(0), torch.tensor(attention_mask).unsqueeze(0)