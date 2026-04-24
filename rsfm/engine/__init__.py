from .train import *
from .eval import *
from .predict import *
from ..dataset import datasets_info


def train_engine_builder(args, cfg, model, optimizer, criterion, trainloader, valloader, trainsampler, valset,
                         rank, logger, writer, previous_best, lr_scheduler, epoch):
    task = datasets_info[cfg['dataset']]['vision_task']

    if task in ['semantic segmentation', 'seg']:
        train_seg(args, cfg, model, optimizer, criterion, trainloader,
                  valloader, trainsampler, rank, logger, writer,
                  previous_best, lr_scheduler, epoch)

    elif task in ['change detection', 'cd']:
        train_cd(args, cfg, model, optimizer, criterion, trainloader,
                 valloader, trainsampler, rank, logger, writer,
                 previous_best, lr_scheduler, epoch)

    elif task in ['scene classification', 'cls']:
        train_cls(args, cfg, model, optimizer, criterion, trainloader,
                  valloader, trainsampler, rank, logger, writer,
                  previous_best, lr_scheduler, epoch)

    elif task in ['referring image segmentation', 'ris']:
        train_ris(args, cfg, model, optimizer, criterion, trainloader,
                  valloader, trainsampler, rank, logger, writer,
                  previous_best, lr_scheduler, epoch)

    elif task in ['visual grounding', 'vg']:
        train_vg(args, cfg, model, optimizer, criterion, trainloader,
                 valloader, trainsampler, rank, logger, writer,
                 previous_best, lr_scheduler, epoch)

    elif task in ['object detection', 'od']:
        train_od(args, cfg, model, optimizer, criterion, trainloader,
                 valloader, trainsampler, rank, logger, writer,
                 previous_best, lr_scheduler, epoch, valset)

    elif task in ['referring expression comprehension and segmentation', 'recs']:
        train_recs(args, cfg, model, optimizer, criterion, trainloader,
                   valloader, trainsampler, rank, logger, writer,
                   previous_best, lr_scheduler, epoch)

    else:
        raise NotImplementedError


def eval_engine_builder(task, model, loader, num_classes, postprocessors=None, base_ds=None, tta=False):
    if task in ['semantic segmentation', 'seg']:
        metrics, metrics_name = eval_seg(model, loader, num_classes, tta)

    elif task in ['change detection', 'cd']:
        metrics, metrics_name = eval_cd(model, loader, num_classes, tta)

    elif task in ['scene classification', 'cls']:
        metrics, metrics_name = eval_cls(model, loader, num_classes, tta)

    elif task in ['referring image segmentation', 'ris']:
        metrics, metrics_name = eval_ris(model, loader, tta)

    elif task in ['visual grounding', 'vg']:
        metrics, metrics_name = eval_vg(model, loader, tta)

    elif task in ['object detection', 'od']:
        metrics, metrics_name = eval_od(model, loader, base_ds, postprocessors, tta)

    elif task in ['referring expression comprehension and segmentation', 'recs']:
        metrics, metrics_name = eval_recs(model, loader, tta)

    else:
        raise NotImplementedError

    return metrics, metrics_name


def predict_engine_builder(task, model, loader, args):
    if task in ['semantic segmentation', 'seg']:
        predict_seg(model, loader, args)

    elif task in ['change detection', 'cd']:
        predict_cd(model, loader, args)

    elif task in ['scene classification', 'cls']:
        predict_cls(model, loader, args)

    elif task in ['referring image segmentation', 'ris']:
        predict_ris(model, loader, args)

    elif task in ['visual grounding', 'vg']:
        predict_vg(model, loader, args)

    elif task in ['referring expression comprehension and segmentation', 'recs']:
        predict_recs(model, loader, args)

    else:
        raise NotImplementedError