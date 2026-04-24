import cv2
import numpy as np
import torch
import torch.nn as nn
from rsfm.foundation_model.sam.v1 import sam_model_registry
from fvcore.nn import FlopCountAnalysis


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

model_type='vit_h'
model = sam_model_registry[model_type](checkpoint='/root/data4/RSFM/pretrained/sam/sam_' + model_type + '.pth')
params = count_parameters(model)
print(f"Parameters: {params / 1e6:.2f}M")

x = torch.randn((1, 3, 1024, 1024))
flops1 = FlopCountAnalysis(model.image_encoder, inputs=(x)).total()
print(f"FLOPs: {flops1 / 1e9:.9f}G")

points = (torch.randn((1, 2, 2)), torch.randn((1, 2)))
flops2 = FlopCountAnalysis(model.prompt_encoder.cpu(), inputs=(points, None, None)).total()

flops3 = FlopCountAnalysis(model.mask_decoder.cpu(),
                           inputs=(torch.randn((1, 256, 64, 64)),
                                   torch.randn((1, 256, 64, 64)),
                                   torch.randn((1, 3, 256)),
                                   torch.randn((1, 256, 64, 64)),
                                   False
                                   )).total()



print(f"FLOPs: {flops2 / 1e9:.9f}G")
print(f"FLOPs: {flops3 / 1e9:.9f}G")
print(f"FLOPs: {(flops1 + flops2 + flops3) / 1e9:.9f}G")