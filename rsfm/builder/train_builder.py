import os
import torch
from utils.utils import count_params
import operator
from functools import reduce
from rsfm.dataset.utils import datasets_info
from rsfm.utils import init_dec_weight, load_from
from .loss import loss_builder
from .optimizer import optimizer_builder
from .scheduler import lr_scheduler_builder, build_scheduler
from .model import seg_model_builder, cd_model_builder, cls_model_builder, \
    ris_model_builder, vg_model_builder, od_model_builder, recs_model_builder



def train_model_builder(cfg, logger, iters_per_epoch):
    task = datasets_info[cfg['dataset']]['vision_task']

    if task in ['semantic segmentation', 'seg']:
        model = seg_model_builder(cfg)

    elif task in ['change detection', 'cd']:
        model = cd_model_builder(cfg)

    elif task in ['scene classification', 'cls']:
        model = cls_model_builder(cfg)

    elif task in ['referring image segmentation', 'ris']:
        model = ris_model_builder(cfg)

    elif task in ['visual grounding', 'vg']:
        model = vg_model_builder(cfg)

    elif task in ['object detection', 'od']:
        model = od_model_builder(cfg)

    elif task in ['referring expression comprehension and segmentation', 'recs']:
        model = recs_model_builder(cfg)

    else:
        raise NotImplementedError

    local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank == 0:
        # logger.info('Total params of visual backbone: {:.2f}M'.format(count_params(model.backbone)))
        # logger.info('Total params of language backbone: {:.2f}M'.format(count_params(model.text_encoder)))
        # logger.info('Total params of neck: {:.2f}M'.format(count_params(model.neck)))
        # logger.info('Total params of decoder: {:.2f}M'.format(count_params(model.decoder)))
        logger.info('Total params: {:.2f}M\n'.format(count_params(model)))

    # load checkpoint of decoder head
    if task != 'scene classification':
        if cfg['model']['decoder'].get('pretrained', False):
            if task == 'visual grounding':
                if cfg['model']['decoder']['kwargs'].get('trans_enc', False):
                    init_dec_weight(cfg, local_rank, logger, model.decoder)
            else:
                init_dec_weight(cfg, local_rank, logger, model.decoder)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      broadcast_buffers=False,
                                                      output_device=local_rank,
                                                      find_unused_parameters=cfg['find_unused_parameters'])

    if cfg['model'].get('load_from', False):
        load_from(cfg, local_rank, logger, model)

    # build specific parameter list to optimize
    if task in ['referring image segmentation', 'ris']:
        single_model = model.module
        backbone_no_decay = list()
        backbone_decay = list()
        for name, m in single_model.backbone.named_parameters():
            if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
                backbone_no_decay.append(m)
            else:
                backbone_decay.append(m)

        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.decoder.parameters() if p.requires_grad]},
            # # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])}
        ]
    elif task in ['visual grounding', 'vg']:
        single_model = model.module
        backbone_param = [p for p in single_model.backbone.parameters() if p.requires_grad]
        vis_enc_param = [p for n, p in single_model.decoder.named_parameters() if p.requires_grad and
                         (n.startswith('backbone_transformer') or n.startswith('input_proj'))]
        bert_param = [p for p in single_model.text_encoder.parameters() if p.requires_grad]
        rest_param = [p for n, p in single_model.decoder.named_parameters() if
                      (p.requires_grad and ('backbone_transformer' not in n) and ('input_proj' not in n))]

        params_to_optimize = [{'params': rest_param, 'lr': cfg['optimizer']['kwargs'].get('lr', 1e-5) *
                                                           cfg['optimizer']['kwargs'].get('lr_multi', 1.0)},
                              {'params': backbone_param, 'lr': cfg['optimizer']['kwargs'].get('lr', 1e-5)},
                              {'params': vis_enc_param, 'lr': cfg['optimizer']['kwargs'].get('lr', 1e-5)},
                              {'params': bert_param, 'lr': cfg['optimizer']['kwargs'].get('lr', 1e-5)}]
    elif task in ['referring expression comprehension and segmentation', 'recs']:
        single_model = model.module
        backbone_no_decay = list()
        backbone_decay = list()
        for name, m in single_model.backbone.named_parameters():
            if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
                backbone_no_decay.append(m)
            else:
                backbone_decay.append(m)

        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.decoder.parameters() if p.requires_grad]},
            {"params": [p for p in single_model.text_encoder.parameters() if p.requires_grad]}
        ]
    else:
        params_to_optimize = None

    optimizer = optimizer_builder(model, cfg['optimizer'], params_to_optimize)

    if task != 'scene classification':
        lr_scheduler = lr_scheduler_builder(optimizer, cfg['lr_scheduler'], iters_per_epoch, cfg['epochs'])
    else:
        lr_scheduler = build_scheduler(optimizer, cfg, iters_per_epoch)

    criterion = loss_builder(cfg['criterion']).cuda(local_rank)

    # model._set_static_graph()

    return model, optimizer, lr_scheduler, criterion