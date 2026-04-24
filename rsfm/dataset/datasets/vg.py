from torch.utils.data import Dataset
from rsfm.dataset.utils import *
from rsfm.dataset.transform.transform import *
from os.path import join as osp
from transformers import RobertaTokenizerFast
from pytorch_pretrained_bert.tokenization import BertTokenizer



class create_vg_dataset(Dataset):
    def __init__(self, cfg, mode='train'):
        self.root, self.size, self.img_suffix, self.mask_suffix, \
        self.num_bands, self.need_bands, self.reduce_zero_label = base_init(cfg)

        self.max_tokens = 20

        self.tokenizer_type = cfg['model']['text_encoder'].get('type', 'bert-base-uncased')
        if self.tokenizer_type == 'bert-base-uncased':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        elif self.tokenizer_type == 'roberta-base':
            self.tokenizer = RobertaTokenizerFast.from_pretrained('pretrained_weights/bert/roberta-base')
        else:
            raise NotImplementedError

        self.mode = mode
        if mode == 'train':
            txt_path = osp(self.root, 'phrase_txts',
                           'trainval.txt' if cfg['dataset'] not in ['RefDIOR_VG', 'RefCOCO_VG', 'RefCOCO+_VG', 'G-Ref-g_VG', 'G-Ref-u_VG']
                           else 'train.json')
            # txt_path = ['/root/lxq/RSFM_RIS_Datasets/Natural/RefCOCO/phrase_txts/train.json',
            #             '/root/lxq/RSFM_RIS_Datasets/Natural/RefCOCO+/phrase_txts/train.json',
            #             '/root/lxq/RSFM_RIS_Datasets/Natural/G-Ref-u/phrase_txts/train.json']
        elif mode == 'val':
            txt_path = osp(self.root, 'phrase_txts',
                           'test.txt' if cfg['dataset'] not in ['RefDIOR_VG', 'RefCOCO_VG', 'RefCOCO+_VG', 'G-Ref-g_VG', 'G-Ref-u_VG']
                           else 'val.json')
        else:
            raise NotImplementedError

        self.uni_ids, self.img_ids, self.bboxs, self.phrases = load_id_phrase_bbox_info(txt_path, self.img_suffix)

        self.transform = make_vlm_transforms(self.size, self.mode, cfg.get('strong_aug', False))

    def __getitem__(self, item):
        img = load_img_mask_file(need_bands=self.need_bands, img_dir=self.img_ids[item], mask_dir=None, need_mask=False)
        bbox = torch.tensor(self.bboxs[item]).float()
        phrase = self.phrases[item].lower()

        target = {}
        target['phrase'] = phrase
        target['bbox'] = bbox

        orig_bbox = bbox.clone()

        img, target = self.transform(img, target)

        if self.tokenizer_type == 'bert-base-uncased':
            examples = read_examples(target['phrase'], item)
            features = convert_examples_to_features(
                examples=examples, seq_length=self.max_tokens, tokenizer=self.tokenizer)
            word_id = torch.tensor(features[0].input_ids, dtype=torch.long)
            word_mask = torch.tensor(features[0].input_mask, dtype=torch.bool)
        else:
            word_id, word_mask = roberta_tokenize(target['phrase'], self.max_tokens, self.tokenizer)

        mask = target.pop('mask')
        bbox = target.pop('bbox')

        if self.mode == 'train':
            return img, mask, word_id, word_mask, bbox
        else:
            size, ratio, orig_size, dxdy = target.pop('size'), target.pop('ratio'), \
                                           target.pop('orig_size'), target.pop('dxdy')

            return img, mask, word_id, word_mask, orig_bbox, size, ratio, orig_size, dxdy, os.path.basename(self.img_ids[item]).split('.')[0]

    def __len__(self):
        return len(self.uni_ids)