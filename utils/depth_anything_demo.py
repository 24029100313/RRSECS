import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose
from rsfm.depth_anything import DepthAnything
from rsfm.depth_anything.transform import Resize, NormalizeImage, PrepareForNet



def estimate(img_path, model_size='base', version='v2', device='cuda'):
    '''
    model_size: small, base, large
    version: v1, v2
    '''
    depth_anything = DepthAnything(model_size)
    depth_anything.to(device)
    if version == 'v1':
        depth_anything.init_weights('pretrained_weights/depth_anything/depth_anything_{:}.pth'.format(model_size))
    else:
        depth_anything.init_weights('pretrained_weights/depth_anything/depth_anything_v2_{:}.pth'.format(model_size))
    depth_anything.eval()

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.0
    h, w, _ = image.shape
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    depth = depth_anything(image) # 1, H, W
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.detach().cpu().numpy().astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    cv2.imwrite(img_path.split('.')[0] + '_depth_' + model_size + '_' + version + '.png', depth)