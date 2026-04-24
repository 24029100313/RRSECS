from addict import Dict
import torch.nn.functional as F
import torch


def init_input_dict(in_index, in_channels, feature_strides):
    input_shape = dict()

    for ind in in_index:
        input_shape['s{}'.format(ind + 1)] = Dict({'channel': in_channels[ind],
                                                   'stride': feature_strides[ind]})

    return input_shape


def semantic_inference(mask_cls, mask_pred):
    mask_cls = F.softmax(mask_cls, dim=-1)[...,1:]
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)

    return semseg


def init_feats_dict(input_shape, features):
    outputs = {}
    for out_ind in input_shape.keys():
        outputs[out_ind] = features[int(out_ind[1:]) - 1]

    return outputs