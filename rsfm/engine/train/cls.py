import torch
from utils.utils import *
from rsfm.dataset import datasets_info
from rsfm.engine.eval import eval_cls
from rsfm.dataset.transform.transform import build_mixup_cutmix
from rsfm.losses import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


# def train_cls(args, cfg, model, optimizer, criterion, trainloader, valloader, trainsampler,
#               rank, logger, writer, previous_best, lr_scheduler, epoch):
#
#     if cfg['mixup_cutmix_aug']:
#         mixup_cutmix_aug = build_mixup_cutmix(cfg)
#     else:
#         mixup_cutmix_aug = None
#
#     for epoch in range(epoch + 1, cfg['epochs']):
#         if rank == 0:
#             logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best Top1 Acc: {:.2f}'.format(
#                 epoch, optimizer.param_groups[0]['lr'], previous_best))
#
#         model.train()
#         total_loss = AverageMeter()
#
#         trainsampler.set_epoch(epoch)
#
#         for i, (img, label) in enumerate(trainloader):
#             img, label = img.cuda(), label.cuda()
#
#             if mixup_cutmix_aug is not None:
#                 img, label = mixup_cutmix_aug(img, label)
#
#             pred = model(img)
#             loss = criterion(pred, label)
#
#             torch.distributed.barrier()
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             lr_scheduler.step()
#
#             total_loss.update(loss.item())
#
#             iters = epoch * len(trainloader) + i
#             if rank == 0:
#                 writer.add_scalar('train/loss', loss.item(), iters)
#
#             if (i % (len(trainloader) // 10) == 0) and (rank == 0):
#                 logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))
#
#         metrics, metrics_name = eval_cls(model, valloader, datasets_info[cfg['dataset']]['num_classes'])
#         # metrics: mAcc1, mAcc5
#
#         if rank == 0:
#             eval_results = metric_table(metrics, metrics_name)
#             logger.info('Eval Results: \n{}'.format(eval_results))
#
#             writer.add_scalar('eval/mAcc1', metrics[0], epoch)
#             writer.add_scalar('eval/mAcc5', metrics[1], epoch)
#
#         is_best = metrics[0] > previous_best
#         previous_best = max(metrics[0], previous_best)
#         if rank == 0:
#             checkpoint = {
#                 'model': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'lr_scheduler': lr_scheduler.state_dict(),
#                 'epoch': epoch,
#                 'previous_best': previous_best,
#             }
#             torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
#             if is_best:
#                 save_best_weight_name = get_save_weight_name(cfg, metrics[0])
#                 delete_previous_best_weight(args.save_path)
#                 torch.save(checkpoint, os.path.join(args.save_path, save_best_weight_name))
#             if cfg.get('save_interval', -1) > 0:
#                 if (epoch + 1) % cfg.get('save_interval') == 0:
#                     torch.save(checkpoint, os.path.join(args.save_path, 'epoch_{}.pth'.format(epoch + 1)))


def train_cls(args, cfg, model, optimizer, criterion, trainloader, valloader,
              trainsampler, rank, logger, writer, previous_best, lr_scheduler, epoch):

    loss_scaler = NativeScalerWithGradNormCount()
    accumulation_steps = cfg.get('accumulation_steps', 1)
    label_smoothing = cfg.get('label_smoothing', 0.1)
    clip_grad = cfg.get('clip_grad', 5.0)
    train_epochs = cfg['epochs']

    if cfg['mixup_cutmix_aug']:
        mixup_cutmix_aug = build_mixup_cutmix(cfg)
    else:
        mixup_cutmix_aug = None

    if cfg['mixup_cutmix_aug']:
        criterion = SoftTargetCrossEntropy()
    elif label_smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epoch + 1, train_epochs):
        trainsampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        num_steps = len(trainloader)

        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        scaler_meter = AverageMeter()

        for idx, (img, label) in enumerate(trainloader):
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            if mixup_cutmix_aug is not None:
                img, label = mixup_cutmix_aug(img, label)

            with torch.cuda.amp.autocast(enabled=True):
                pred = model(img)
            
            loss = criterion(pred, label)
            loss = loss / accumulation_steps

            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=clip_grad,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(idx + 1) % accumulation_steps == 0)

            if (idx + 1) % accumulation_steps == 0:
                optimizer.zero_grad()
                lr_scheduler.step_update((epoch * num_steps + idx) // accumulation_steps)
            loss_scale_value = loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()

            loss_meter.update(loss.item(), label.size(0))
            if grad_norm is not None:  # loss_scaler return None if not update
                norm_meter.update(grad_norm)
            scaler_meter.update(loss_scale_value)

            iters = epoch * num_steps + idx
            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)

            if (idx % 100 == 0) and (rank == 0):
                lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f'Train: [{epoch + 1}/{train_epochs}][{idx}/{num_steps}]\t'
                    f'lr {lr:.6f}\t '
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})')

        metrics, metrics_name = eval_cls(model, valloader, datasets_info[cfg['dataset']]['num_classes'])
        # metrics: mAcc1, mAcc5

        if rank == 0:
            eval_results = metric_table(metrics, metrics_name)
            logger.info('Eval Results: \n{}'.format(eval_results))

            writer.add_scalar('eval/mAcc1', metrics[0], epoch)
            writer.add_scalar('eval/mAcc5', metrics[1], epoch)

        is_best = metrics[0] > previous_best
        previous_best = max(metrics[0], previous_best)

        if rank == 0:
            logger.info(f'Current Epoch: {epoch + 1}\t'
                        f'Current best Top1 Acc: {metrics[0]:.2f}\t'
                        f'Previous best Top1 Acc: {previous_best:.2f}')

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                save_best_weight_name = get_save_weight_name(cfg, metrics[0])
                delete_previous_best_weight(args.save_path)
                torch.save(checkpoint, os.path.join(args.save_path, save_best_weight_name))
            if cfg.get('save_interval', -1) > 0:
                if (epoch + 1) % cfg.get('save_interval') == 0:
                    torch.save(checkpoint, os.path.join(args.save_path, 'epoch_{}.pth'.format(epoch + 1)))