import numpy as np
import random
import torch
from PIL import ImageFilter, ImageEnhance
import torchvision.transforms as T
import torchvision.transforms.functional as F
from utils.bbox_utils import xyxy2xywh, box_iou



class RandomBrightness(object):
    def __init__(self, brightness=0.4):
        assert brightness >= 0.0
        assert brightness <= 1.0
        self.brightness = brightness

    def __call__(self, img):
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        return img


class RandomContrast(object):
    def __init__(self, contrast=0.4):
        assert contrast >= 0.0
        assert contrast <= 1.0
        self.contrast = contrast

    def __call__(self, img):
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        return img


class RandomSaturation(object):
    def __init__(self, saturation=0.4):
        assert saturation >= 0.0
        assert saturation <= 1.0
        self.saturation = saturation

    def __call__(self, img):
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
        return img


class ColorJitter(object):
    def __init__(self, p=0.8, brightness=0.4, contrast=0.4, saturation=0.4):
        self.p = p
        self.rand_brightness = RandomBrightness(brightness)
        self.rand_contrast = RandomContrast(contrast)
        self.rand_saturation = RandomSaturation(saturation)

    def __call__(self, img, target=None):
        if random.random() < self.p:
            func_inds = list(np.random.permutation(3))
            for func_id in func_inds:
                if func_id == 0:
                    img = self.rand_brightness(img)
                elif func_id == 1:
                    img = self.rand_contrast(img)
                elif func_id == 2:
                    img = self.rand_saturation(img)

        if target is None:
            return img
        return img, target


class GaussianBlur(object):
    def __init__(self, p=0.5, sigma=[.1, 2.]):
        self.sigma = sigma
        self.p = p

    def __call__(self, img, target=None):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

        if target is None:
            return img
        return img, target


class RandomGray(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img, target=None):
        img = T.RandomGrayscale(self.p)(img)

        if target is None:
            return img
        return img, target


class EdgeEnhance(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target=None):
        if random.random() < self.p:
            img = img.filter(ImageFilter.EDGE_ENHANCE)

        if target is None:
            return img
        return img, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.hflip(img)
            w, h = img.size

            if 'ris_mask' in target and target['ris_mask'] is not None:
                target['ris_mask'] = F.hflip(target['ris_mask'])

            if 'bbox' in target and target['bbox'] is not None:
                target['bbox'] = target['bbox'][[2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])

            target['phrase'] = target['phrase'].replace('right', '*&^special^&*').replace('left', 'right').replace('*&^special^&*', 'left')

        return img, target


class ToTensor(object):
    def __call__(self, img, target=None):
        if target is None:
            return F.to_tensor(img)
        return F.to_tensor(img), target


class RandomResize(object):
    def __init__(self, sizes, resize_long_side=True, record_resize_info=False):
        self.sizes = sizes
        self.resize_long_side = resize_long_side
        if resize_long_side:
            self.choose_size = max
        else:
            self.choose_size = min
        self.record_resize_info = record_resize_info

    def __call__(self, img, target):
        size = random.choice(self.sizes)
        h, w = img.height, img.width
        ratio = float(size) / self.choose_size(h, w)
        new_h, new_w = round(h * ratio), round(w * ratio)
        img = F.resize(img, (new_h, new_w))

        if 'ris_mask' in target and target['ris_mask'] is not None:
            target['ris_mask'] = F.resize(target['ris_mask'], (new_h, new_w), interpolation=F.InterpolationMode.NEAREST)

        ratio_h, ratio_w = float(new_h) / h, float(new_w) / w

        if 'bbox' in target and target['bbox'] is not None:
            target['bbox'] = target['bbox'] * torch.as_tensor([ratio_w, ratio_h, ratio_w, ratio_h])

        if self.record_resize_info:
            target['orig_size'] = torch.as_tensor([h, w], dtype=torch.float32)
            target['ratio'] = torch.as_tensor([ratio_h, ratio_w], dtype=torch.float32)
            target['size'] = torch.as_tensor([new_h, new_w], dtype=torch.float32)

        return img, target


def crop(image, box, region):
    cropped_image = F.crop(image, *region)

    i, j, h, w = region

    max_size = torch.as_tensor([w, h], dtype=torch.float32)
    cropped_box = box - torch.as_tensor([j, i, j, i])
    cropped_box = torch.min(cropped_box.reshape(2, 2), max_size)
    cropped_box = cropped_box.clamp(min=0)
    cropped_box = cropped_box.reshape(-1)

    return cropped_image, cropped_box


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, max_try: int = 20):
        self.min_size = min_size
        self.max_size = max_size
        self.max_try = max_try

    def __call__(self, img, target):
        box = target['bbox']

        num_try = 0
        while num_try < self.max_try:
            num_try += 1
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            region = T.RandomCrop.get_params(img, [h, w])  # [i, j, target_w, target_h]
            box_xywh = xyxy2xywh(box)
            box_x, box_y = box_xywh[0], box_xywh[1]
            if box_x > region[0] and box_y > region[1]:
                img, box = crop(img, box, region)
                target['bbox'] = box
                return img, target

        return img, target


class RandomSelect(object):
    def __init__(self, transforms1, transforms2, p=0.5,
                 exclude_words=['left', 'right', 'top', 'bottom', 'middle']):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p
        self.exclude_words = exclude_words

    def __call__(self, img, target):
        phrase = target['phrase']
        for word in self.exclude_words:
            if word in phrase:
                return self.transforms1(img, target)

        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        if target is None:
            for t in self.transforms:
                image = t(image)
            return image

        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class NormalizeAndPad(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=640,
                 aug_translate=False, center_place=False, padding=True):
        self.mean = mean
        self.std = std
        self.size = size
        self.aug_translate = aug_translate
        self.center_place = center_place
        self.padding = padding

    def __call__(self, img, target=None):
        img = F.normalize(img, mean=self.mean, std=self.std)

        h, w = img.shape[1:]
        dw = self.size - w
        dh = self.size - h

        if self.aug_translate:
            top = random.randint(0, dh)
            left = random.randint(0, dw)
        elif self.center_place:
            top = round(dh / 2.0 - 0.1)
            left = round(dw / 2.0 - 0.1)
        else:
            top = left = 0

        if target is None:
            if self.padding:
                out_img = torch.zeros((3, self.size, self.size)).float()
                out_img[:, top:top + h, left:left + w] = img
            else:
                out_img = img
            return out_img

        if self.padding:
            out_img = torch.zeros((3, self.size, self.size)).float()
            out_img[:, top:top + h, left:left + w] = img
            out_mask = torch.zeros((self.size, self.size), dtype=torch.bool)
            target['mask'] = out_mask

            if 'ris_mask' in target and target['ris_mask'] is not None:
                ris_mask = torch.from_numpy(np.array(target['ris_mask'])).long()
                out_ris_mask = torch.zeros((self.size, self.size)).long()
                out_ris_mask[top:top + h, left:left + w] = ris_mask
                target['ris_mask'] = out_ris_mask
        else:
            out_img = img

        out_h, out_w = out_img.shape[-2:]
        target['size'] = torch.as_tensor([out_h, out_w], dtype=torch.float32)
        target['dxdy'] = torch.as_tensor([left, top], dtype=torch.float32)

        if 'bbox' in target and target['bbox'] is not None:
            box = target['bbox']
            box[0], box[2] = box[0] + left, box[2] + left
            box[1], box[3] = box[1] + top, box[3] + top
            box = xyxy2xywh(box)
            box = box / torch.tensor([out_w, out_h, out_w, out_h], dtype=torch.float32)
            target['bbox'] = box

        return out_img, target