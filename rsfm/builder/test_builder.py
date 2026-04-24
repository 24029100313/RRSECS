import os, yaml
import torch
from utils.utils import count_params
from rsfm.dataset.utils import datasets_info
from pathlib import Path
from .model import seg_model_builder, cd_model_builder, cls_model_builder, \
    ris_model_builder, vg_model_builder, od_model_builder, recs_model_builder



# def test_model_builder(args):
#     dataset_name, backbone, neck, decoder, _, img_size = os.path.basename(args.weight_path).split('.')[:6]
#     training_size = int(img_size[3:])
#
#     vision_task = datasets_info[dataset_name]['vision_task']
#     num_classes = datasets_info[dataset_name]['num_classes']
#
#     if vision_task in ['referring image segmentation', 'visual grounding', 'recs']:
#         text_cfg = {'type': 'bert-base-uncased'}
#     else:
#         text_cfg = None
#
#     # build config
#     cfg = {'model':
#                {'backbone':
#                     {'type': backbone,
#                      'pretrained': None,
#                      'kwargs':
#                          {'in_channels': 3,
#                           'vlf_ris': decoder.replace('Head', '') if vision_task in ['referring image segmentation', 'recs'] else None,
#                           # 'vlf_ris': None,
#                           'vlf_vg': decoder.replace('Head', '') if vision_task == 'visual grounding'
#                                                                    and decoder in ['QRNetHead', 'LPVAHead'] else None}
#                      },
#                 'text_encoder': text_cfg
#                 },
#            'dataset': dataset_name,
#            'crop_size': training_size,
#            'criterion': {'kwargs': {}}}
#     if neck != 'None':
#         cfg['model']['neck'] = {'type': neck}
#     if decoder != 'None':
#         cfg['model']['decoder'] = {'type': decoder,
#                                    'kwargs': {'trans_enc': False,
#                                               # 'num_queries': 10
#                                               'num_classes': 1
#                                               }}
#
#     if vision_task in ['semantic segmentation', 'seg']:
#         model = seg_model_builder(cfg)
#
#     elif vision_task in ['change detection', 'cd']:
#         model = cd_model_builder(cfg)
#
#     elif vision_task in ['scene classification', 'cls']:
#         model = cls_model_builder(cfg)
#
#     elif vision_task in ['referring image segmentation', 'ris']:
#         model = ris_model_builder(cfg)
#
#     elif vision_task in ['visual grounding', 'vg']:
#         model = vg_model_builder(cfg)
#
#     elif vision_task in ['object detection', 'od']:
#         model = od_model_builder(cfg)
#
#     elif vision_task in ['referring expression comprehension and segmentation', 'recs']:
#         model = recs_model_builder(cfg)
#
#     else:
#         raise NotImplementedError
#
#     local_rank = int(os.environ["LOCAL_RANK"])
#
#     if local_rank == 0:
#         print('Loading from {}\n'.format(args.weight_path))
#         print('Total params: {:.2f}M\n'.format(count_params(model)))
#
#     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#     model.cuda(local_rank)
#     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
#                                                       broadcast_buffers=False,
#                                                       output_device=local_rank)
#     checkpoint = torch.load(args.weight_path)
#     model.load_state_dict(checkpoint['model'])
#
#     return model, vision_task, num_classes, dataset_name, training_size



def test_model_builder(args):
    root_path = args.output_path
    config_path = [p for p in Path(root_path).rglob('*.yaml') if p.is_file()][0]
    weight_path = [p for p in Path(root_path).rglob('*best*.pth') if p.is_file()][0]
    # weight_path = [p for p in Path(root_path).rglob('*epoch_30*.pth') if p.is_file()][0]

    cfg = yaml.load(open(config_path, "r"), Loader=yaml.Loader)

    dataset_name = cfg['dataset']
    vision_task = datasets_info[dataset_name]['vision_task']
    num_classes = datasets_info[dataset_name]['num_classes']
    training_size = int(cfg['crop_size'])

    if vision_task in ['semantic segmentation', 'seg']:
        model = seg_model_builder(cfg)
    elif vision_task in ['change detection', 'cd']:
        model = cd_model_builder(cfg)
    elif vision_task in ['scene classification', 'cls']:
        model = cls_model_builder(cfg)
    elif vision_task in ['referring image segmentation', 'ris']:
        model = ris_model_builder(cfg)
    elif vision_task in ['visual grounding', 'vg']:
        model = vg_model_builder(cfg)
    elif vision_task in ['object detection', 'od']:
        model = od_model_builder(cfg)
    elif vision_task in ['referring expression comprehension and segmentation', 'recs']:
        model = recs_model_builder(cfg)
    else:
        raise NotImplementedError

    local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank == 0:
        print('Loading from {}\n'.format(weight_path))
        print('Total params: {:.2f}M\n'.format(count_params(model)))

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      broadcast_buffers=False,
                                                      output_device=local_rank)

    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['model'])

    return model, vision_task, num_classes, dataset_name, training_size