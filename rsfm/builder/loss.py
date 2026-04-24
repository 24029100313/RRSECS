import numpy as np
import torch
import torch.nn as nn
import importlib


def loss_builder(criterion_cfg):
    if criterion_cfg['type'] == 'CELoss':
        return nn.CrossEntropyLoss(**criterion_cfg['kwargs'])
        # return nn.CrossEntropyLoss(weight=weights, **criterion_cfg['kwargs'])
    else:
        criterion = getattr(importlib.import_module('rsfm.losses'), criterion_cfg['type'])
        return criterion(**criterion_cfg['kwargs'] if criterion_cfg['kwargs'] else {})