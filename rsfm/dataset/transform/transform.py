import math
import random
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from timm.data import create_transform, Mixup
from rsfm.dataset.utils import datasets_info
import rsfm.dataset.transform.transform_vlm as TVLM
import rsfm.dataset.transform.transform_dino as TOD


inter_dict = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC, 'nearest': Image.NEAREST}


def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    if mask is not None:
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    if mask is not None:
        mask = mask.crop((x, y, x + size, y + size))
        return img, mask

    return img


def crop2(img1, img2, mask, size, ignore_value=255):
    w, h = img1.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img1 = ImageOps.expand(img1, border=(0, 0, padw, padh), fill=0)
    img2 = ImageOps.expand(img2, border=(0, 0, padw, padh), fill=0)
    if mask is not None:
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img1.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img1 = img1.crop((x, y, x + size, y + size))
    img2 = img2.crop((x, y, x + size, y + size))
    if mask is not None:
        mask = mask.crop((x, y, x + size, y + size))
        return img1, img2, mask

    return img1, img2


def hflip(img, mask=None, p=0.5):
    if mask is not None:
        if random.random() < p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask
    else:
        if random.random() < p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


def hflip2(img1, img2, mask=None, p=0.5):
    if mask is not None:
        if random.random() < p:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img1, img2, mask
    else:
        if random.random() < p:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
        return img1, img2


def vflip(img, mask=None, p=0.5):
    if mask is not None:
        if random.random() < p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask
    else:
        if random.random() < p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


def vflip2(img1, img2, mask=None, p=0.5):
    if mask is not None:
        if random.random() < p:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img1, img2, mask
    else:
        if random.random() < p:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
        return img1, img2


def extend_list(original_list, num_bands):
    extended_list = original_list.copy()
    original_len = len(original_list)
    if num_bands > original_len:
        extension = [original_list[i % original_len] for i in range(original_len, num_bands)]
        extended_list.extend(extension)
    return extended_list


def normalize(img, mask=None, num_bands=3):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(extend_list([0.485, 0.456, 0.406], num_bands),
                             extend_list([0.229, 0.224, 0.225], num_bands)),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def normalize2(img1, img2, mask=None, num_bands=3):
    img1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(extend_list([0.485, 0.456, 0.406], num_bands),
                             extend_list([0.229, 0.224, 0.225], num_bands)),
    ])(img1)
    img2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(extend_list([0.485, 0.456, 0.406], num_bands),
                             extend_list([0.229, 0.224, 0.225], num_bands)),
    ])(img2)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img1, img2, mask
    return img1, img2


def resize_and_pad(img, mask, base_size, interpolation='bilinear'):
    w, h = img.size
    scale = base_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    img_resized = img.resize((new_w, new_h), inter_dict[interpolation])
    mask_resized = mask.resize((new_w, new_h), Image.NEAREST)

    img_padded = Image.new('RGB', (base_size, base_size), color=128)
    mask_padded = Image.new('L', (base_size, base_size), color=255)

    img_padded.paste(img_resized, ((base_size - new_h) // 2, (base_size - new_w) // 2))
    mask_padded.paste(mask_resized, ((base_size - new_h) // 2, (base_size - new_w) // 2))

    return img_padded, mask_padded


def resize(img, mask, base_size, ratio_range, interpolation='bilinear'):
    w, h = img.size
    long_side = random.randint(int(base_size * ratio_range[0]), int(base_size * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), inter_dict[interpolation])

    if mask is not None:
        mask = mask.resize((ow, oh), Image.NEAREST)
        return img, mask

    return img


def resize2(img1, img2, mask, base_size, ratio_range, interpolation='bilinear'):
    w, h = img1.size
    long_side = random.randint(int(base_size * ratio_range[0]), int(base_size * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img1 = img1.resize((ow, oh), inter_dict[interpolation])
    img2 = img2.resize((ow, oh), inter_dict[interpolation])

    if mask is not None:
        mask = mask.resize((ow, oh), Image.NEAREST)
        return img1, img2, mask

    return img1, img2


def resize_fixed(img, mask, size, interpolation='bilinear'):
    img = img.resize((size, size), inter_dict[interpolation])
    if mask is not None:
        mask = mask.resize((size, size), Image.NEAREST)
        return img, mask

    return img


def resize_pct_center(img, size, pct=0.875, interpolation='bilinear'):
    new_size = math.floor(size / pct)
    img = img.resize((new_size, new_size), inter_dict[interpolation])

    x = y = (new_size - size) // 2
    img = img.crop((x, y, x + size, y + size))

    return img


def resize2_fixed(img1, img2, mask, size, interpolation='bilinear'):
    img1 = img1.resize((size, size), inter_dict[interpolation])
    img2 = img2.resize((size, size), inter_dict[interpolation])
    if mask is not None:
        mask = mask.resize((size, size), Image.NEAREST)
        return img1, img2, mask

    return img1, img2


def build_transform(cfg):
    transform = create_transform(input_size=cfg['crop_size'],
                                 is_training=True,
                                 color_jitter=0.4,
                                 auto_augment='rand-m9-mstd0.5-inc1',
                                 re_prob=0.25,
                                 re_mode='pixel',
                                 re_count=1,
                                 interpolation='bicubic')

    return transform


def build_mixup_cutmix(cfg):
    mixup_cutmix_aug = Mixup(mixup_alpha=0.8,
                             cutmix_alpha=1.0,
                             cutmix_minmax=None,
                             prob=1.0,
                             switch_prob=0.5,
                             mode='batch',
                             label_smoothing=0.1,
                             num_classes=datasets_info[cfg['dataset']]['num_classes'])

    return mixup_cutmix_aug


def make_vlm_transforms(img_size=640, split='train', strong_aug=False):
    if split == 'train':
        if strong_aug:
            scales = [img_size - (32 * i) for i in range(6, -1, -1)]
            return TVLM.Compose([
                TVLM.RandomResize(scales),
                TVLM.ColorJitter(),
                TVLM.RandomHorizontalFlip(),
                TVLM.ToTensor(),
                TVLM.NormalizeAndPad(size=img_size, aug_translate=True)
            ])
        else:
            return TVLM.Compose([
                TVLM.RandomResize([img_size]),
                TVLM.ToTensor(),
                TVLM.NormalizeAndPad(size=img_size, center_place=True)
            ])

    elif split in ['val', 'test']:
        return TVLM.Compose([
            TVLM.RandomResize([img_size], record_resize_info=True),
            TVLM.ToTensor(),
            TVLM.NormalizeAndPad(size=img_size, center_place=True)
        ])

    else:
        raise ValueError(f'unknown {split}')


def make_od_transforms(cfg, mode='train'):
    normalize = TOD.Compose([
        TOD.ToTensor(),
        TOD.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]

    # update args from config files
    scales = cfg.get('scales', scales)
    max_size = cfg.get('max_size', max_size)
    scales2_resize = cfg.get('scales2_resize', scales2_resize)
    scales2_crop = cfg.get('scales2_crop', scales2_crop)

    if mode == 'train':
        if cfg.get('fix_size', False):
            return TOD.Compose([
                TOD.RandomHorizontalFlip(),
                TOD.RandomResize([(max_size, max(scales))]),
                normalize,
            ])

        if cfg.get('strong_aug', False):
            import datasets.sltransform as SLT

            return TOD.Compose([
                TOD.RandomHorizontalFlip(),
                TOD.RandomSelect(
                    TOD.RandomResize(scales, max_size=max_size),
                    TOD.Compose([
                        TOD.RandomResize(scales2_resize),
                        TOD.RandomSizeCrop(*scales2_crop),
                        TOD.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    # SLT.Rotate(10),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),
                normalize,
            ])

        return TOD.Compose([
            TOD.RandomHorizontalFlip(),
            TOD.RandomSelect(
                TOD.RandomResize(scales, max_size=max_size),
                TOD.Compose([
                    TOD.RandomResize(scales2_resize),
                    TOD.RandomSizeCrop(*scales2_crop),
                    TOD.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if mode in ['val', 'test']:
        return TOD.Compose([
            TOD.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {mode}')