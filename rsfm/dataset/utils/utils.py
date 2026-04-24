import json
import re, os
import torch, cv2
import numpy as np
from PIL import Image
import skimage.io as io
from typing import Optional, List
from torch import Tensor
import torch.nn.functional as F
from rsfm.dataset.datasets_info import datasets_info



def mask_preprocess(mask, reduce_zero_label=False, convert_255_1=False):
    if reduce_zero_label:
        mask[mask == 0] = 255
        mask -= 1
        mask[mask == 254] = 255

        return mask

    if convert_255_1:
        # assert np.all(mask[mask != 0] == 255)
        mask[mask != 0] = 1

        return mask

    return mask


def min_max_normalize(mask):
    normalized_image = (mask - np.min(mask)) / (np.max(mask) - np.min(mask) + 1e-10) * 255
    result = normalized_image.astype(np.uint8)
    return result


def base_init(cfg):
    dataset_info = datasets_info.get(cfg['dataset'])

    root = cfg.get('data_root', dataset_info.get('data_root'))
    size = cfg.get('crop_size', dataset_info.get('training_size'))
    img_suffix = cfg.get('img_suffix', dataset_info.get('img_suffix'))
    mask_suffix = cfg.get('mask_suffix', dataset_info.get('mask_suffix'))
    num_bands = cfg['model']['backbone']['kwargs']['in_channels']
    need_bands = cfg.get('need_bands', dataset_info.get('need_bands'))
    reduce_zero_label = cfg.get('reduce_zero_label', dataset_info.get('reduce_zero_label'))
    convert_255_1 = cfg.get('convert_255_1', dataset_info.get('convert_255_1', False))

    return root, size, img_suffix, mask_suffix, num_bands, need_bands, reduce_zero_label, convert_255_1


def load_img_mask_file(need_bands, img_dir, mask_dir,
                       need_mask=True, reduce_zero_label=False, convert_255_1=False):
    if need_bands:
        raw_img = io.imread(img_dir)
        selected_bands = [raw_img[:, :, band][:, :, np.newaxis] for band in need_bands]
        if len(selected_bands) == 1:
            img = Image.fromarray(np.repeat(min_max_normalize(selected_bands[0]), 3, axis=2))
        else:
            img = Image.fromarray(min_max_normalize(np.concatenate(selected_bands, axis=-1)))
    else: # channel=1,3
        img = Image.fromarray(cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB))
        # img = Image.fromarray(cv2.imread(img_dir))

    if need_mask:
        mask = Image.fromarray(mask_preprocess(cv2.imread(mask_dir, 0), reduce_zero_label, convert_255_1))

        return img, mask

    return img


def load_img1_img2_mask_file(need_bands, img1_dir, img2_dir, mask_dir,
                             need_mask=True, reduce_zero_label=False, convert_255_1=False):
    if need_bands:
        raw_img1 = io.imread(img1_dir)
        selected_bands1 = [raw_img1[:, :, band][:, :, np.newaxis] for band in need_bands]
        if len(selected_bands1) == 1:
            img1 = Image.fromarray(np.repeat(min_max_normalize(selected_bands1[0]), 3, axis=2))
        else:
            img1 = Image.fromarray(min_max_normalize(np.concatenate(selected_bands1, axis=-1)))

        raw_img2 = io.imread(img2_dir)
        selected_bands2 = [raw_img2[:, :, band][:, :, np.newaxis] for band in need_bands]
        if len(selected_bands2) == 1:
            img2 = Image.fromarray(np.repeat(min_max_normalize(selected_bands2[0]), 3, axis=2))
        else:
            img2 = Image.fromarray(min_max_normalize(np.concatenate(selected_bands2, axis=-1)))
    else: # channel=1,3
        img1 = Image.fromarray(cv2.cvtColor(cv2.imread(img1_dir), cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(cv2.imread(img2_dir), cv2.COLOR_BGR2RGB))

    if need_mask:
        mask = Image.fromarray(mask_preprocess(cv2.imread(mask_dir, 0), reduce_zero_label, convert_255_1))

        return img1, img2, mask

    return img1, img2


def load_id_phrase_mask_info(root, txt_path, img_suffix, mask_suffix, img_text_single=False):
    def _load_single_file(root, path, img_suffix, mask_suffix):
        uni_ids, img_ids, mask_ids, phrases = [], [], [], []

        if path.endswith('.txt'):
            for uni_id, line in enumerate(open(path, 'r').readlines()):
                uni_ids.append(uni_id)
                info = line.strip().split(' ')
                img_ids.append(root + '/images/' + info[0] + img_suffix)
                mask_ids.append(root + '/masks/' + info[0] + mask_suffix)
                phrases.append(' '.join(i for i in info[1:]))
        elif path.endswith('.json'):
            ind = 0
            for k, v in json.load(open(path, 'r')).items():
                if img_text_single:
                    uni_ids.append(ind)
                    img_ids.append(root + '/images/' + k + img_suffix)
                    mask_ids.append(root + '/masks/' + k + mask_suffix)
                    phrases.append(np.random.choice(v['sentences']))
                    ind += 1
                else:
                    for sent in v['sentences']:
                        uni_ids.append(ind)
                        img_ids.append(root + '/images/' + k + img_suffix)
                        mask_ids.append(root + '/masks/' + k + mask_suffix)
                        phrases.append(sent)
                        ind += 1
        else:
            raise NotImplementedError

        return uni_ids, img_ids, mask_ids, phrases

    if isinstance(txt_path, list):
        uni_ids, img_ids, mask_ids, phrases = [], [], [], []
        for path in txt_path:
            single_uni_ids, single_img_ids, single_mask_ids, single_phrases = \
                _load_single_file(root, path, img_suffix, mask_suffix)
            uni_ids += single_uni_ids
            img_ids += single_img_ids
            mask_ids += single_mask_ids
            phrases += single_phrases
    else:
        uni_ids, img_ids, mask_ids, phrases = _load_single_file(root, txt_path, img_suffix, mask_suffix)

    return uni_ids, img_ids, mask_ids, phrases


def load_id_phrase_bbox_info(txt_path, img_suffix, img_text_single=False):
    def _load_single_file(path, suffix):
        uni_ids, img_ids, bboxs, phrases = [], [], [], []
        root = os.path.dirname(os.path.dirname(path))

        if path.endswith('.txt'):
            for uni_id, line in enumerate(open(path, 'r').readlines()):
                uni_ids.append(uni_id)
                info = line.strip().split(' ')
                img_ids.append(root + '/images/' + info[0] + suffix)
                bboxs.append(np.array([int(info[1]), int(info[2]), int(info[3]), int(info[4])]))
                phrases.append(' '.join(i for i in info[5:]))
        elif path.endswith('.json'):
            ind = 0
            for k, v in json.load(open(path, 'r')).items():
                img_id, bbox = k, v['bbox']
                if img_text_single:
                    uni_ids.append(ind)
                    img_ids.append(root + '/images/' + img_id + suffix)
                    bboxs.append(bbox)
                    phrases.append(np.random.choice(v['sentences']))
                    ind += 1
                else:
                    for sent in v['sentences']:
                        uni_ids.append(ind)
                        img_ids.append(root + '/images/' + img_id + suffix)
                        bboxs.append(bbox)
                        phrases.append(sent)
                        ind += 1
        else:
            raise NotImplementedError

        return uni_ids, img_ids, bboxs, phrases

    if isinstance(txt_path, list):
        uni_ids, img_ids, bboxs, phrases = [], [], [], []
        for path in txt_path:
            single_uni_ids, single_img_ids, single_bboxs, single_phrases = _load_single_file(path, img_suffix)
            uni_ids += single_uni_ids
            img_ids += single_img_ids
            bboxs += single_bboxs
            phrases += single_phrases
    else:
        uni_ids, img_ids, bboxs, phrases = _load_single_file(txt_path, img_suffix)

    return uni_ids, img_ids, bboxs, phrases


def load_id_phrase_mask_bbox_info(root, txt_path, img_suffix, mask_suffix, img_text_single=False):
    def _load_single_file(root, path, img_suffix, mask_suffix):
        uni_ids, img_ids, mask_ids, bboxs, phrases = [], [], [], [], []

        if path.endswith('.txt'):
            for uni_id, line in enumerate(open(path, 'r').readlines()):
                uni_ids.append(uni_id)
                info = line.strip().split(' ')
                img_ids.append(root + '/images/' + info[0] + img_suffix)
                mask_ids.append(root + '/masks/' + info[0] + mask_suffix)
                bboxs.append(np.array([int(info[1]), int(info[2]), int(info[3]), int(info[4])]))
                phrases.append(' '.join(i for i in info[5:]))
        elif path.endswith('.json'):
            ind = 0
            for k, v in json.load(open(path, 'r')).items():
                if img_text_single:
                    uni_ids.append(ind)
                    img_ids.append(root + '/images/' + k + img_suffix)
                    mask_ids.append(root + '/masks/' + k + mask_suffix)
                    bboxs.append(v['bbox'])
                    phrases.append(np.random.choice(v['sentences']))
                    ind += 1
                else:
                    for sent in v['sentences']:
                        uni_ids.append(ind)
                        img_ids.append(root + '/images/' + k + img_suffix)
                        mask_ids.append(root + '/masks/' + k + mask_suffix)
                        bboxs.append(v['bbox'])
                        phrases.append(sent)
                        ind += 1
        else:
            raise NotImplementedError

        return uni_ids, img_ids, mask_ids, bboxs, phrases

    if isinstance(txt_path, list):
        uni_ids, img_ids, mask_ids, bboxs, phrases = [], [], [], [], []
        for path in txt_path:
            single_uni_ids, single_img_ids, single_mask_ids, single_bboxs, single_phrases =\
                _load_single_file(root, path, img_suffix, mask_suffix)
            uni_ids += single_uni_ids
            img_ids += single_img_ids
            mask_ids += single_mask_ids
            bboxs += single_bboxs
            phrases += single_phrases
    else:
        uni_ids, img_ids, mask_ids, bboxs, phrases = _load_single_file(root, txt_path, img_suffix, mask_suffix)

    return uni_ids, img_ids, mask_ids, bboxs, phrases


# vg language utils
def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line  # reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples


## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def roberta_tokenize(phrase, seq_length, tokenizer):
    tokenized = tokenizer.batch_encode_plus([phrase], max_length=seq_length, return_tensors="pt", truncation=True)
    word_id, word_mask = tokenized.input_ids, tokenized.attention_mask

    token_length = word_id.size(1)
    if token_length < seq_length:
        word_id = pad_tensor(word_id, seq_length, token_length)
        word_mask = pad_tensor(word_mask, seq_length, token_length)

    return word_id, word_mask


def pad_tensor(orig_tensor, seq_length, token_length):
    padding = (0, seq_length - token_length)
    expanded_tensor = F.pad(orig_tensor, padding, mode='constant', value=0)

    return expanded_tensor


def collate_fn(batch):
    # import ipdb; ipdb.set_trace()
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
        if mask == 'auto':
            self.mask = torch.zeros_like(tensors).to(tensors.device)
            if self.mask.dim() == 3:
                self.mask = self.mask.sum(0).to(bool)
            elif self.mask.dim() == 4:
                self.mask = self.mask.sum(1).to(bool)
            else:
                raise ValueError("tensors dim must be 3 or 4 but {}({})".format(self.tensors.dim(), self.tensors.shape))

    def imgsize(self):
        res = []
        for i in range(self.tensors.shape[0]):
            mask = self.mask[i]
            maxH = (~mask).sum(0).max()
            maxW = (~mask).sum(1).max()
            res.append(torch.Tensor([maxH, maxW]))
        return res

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def to_img_list_single(self, tensor, mask):
        assert tensor.dim() == 3, "dim of tensor should be 3 but {}".format(tensor.dim())
        maxH = (~mask).sum(0).max()
        maxW = (~mask).sum(1).max()
        img = tensor[:, :maxH, :maxW]
        return img

    def to_img_list(self):
        """remove the padding and convert to img list

        Returns:
            [type]: [description]
        """
        if self.tensors.dim() == 3:
            return self.to_img_list_single(self.tensors, self.mask)
        else:
            res = []
            for i in range(self.tensors.shape[0]):
                tensor_i = self.tensors[i]
                mask_i = self.mask[i]
                res.append(self.to_img_list_single(tensor_i, mask_i))
            return res

    @property
    def device(self):
        return self.tensors.device

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    @property
    def shape(self):
        return {
            'tensors.shape': self.tensors.shape,
            'mask.shape': self.mask.shape
        }


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')

    return NestedTensor(tensor, mask)
