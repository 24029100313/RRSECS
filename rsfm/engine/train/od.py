from utils.utils import *
from rsfm.engine.eval import eval_od
from rsfm.utils import DETRPostProcess, DINOPostProcess



def train_od(args, cfg, model, optimizer, criterion, trainloader, valloader, trainsampler,
             rank, logger, writer, previous_best, lr_scheduler, epoch, valset):
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.7f}, Previous best mAP: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = AverageMeter()
        loss_bbox = AverageMeter()
        loss_giou = AverageMeter()
        loss_cls = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (samples, targets) in enumerate(trainloader):
            samples = samples.to(torch.device('cuda'))
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            outputs = model(samples, targets) # for dn training

            loss_dict = criterion(outputs, targets)

            optimizer.zero_grad()
            loss_dict['total_loss'].backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()

            total_loss.update(loss_dict['total_loss'].item())
            loss_bbox.update(loss_dict['loss_bbox'].item())
            loss_giou.update(loss_dict['loss_giou'].item())
            if 'loss_cls' in loss_dict.keys():
                loss_cls.update(loss_dict['loss_cls'].item())

            iters = epoch * len(trainloader) + i
            if rank == 0:
                writer.add_scalar('train/loss', loss_dict['total_loss'].item(), iters)
                writer.add_scalar('train/loss_bbox', loss_dict['loss_bbox'].item(), iters)
                writer.add_scalar('train/loss_giou', loss_dict['loss_giou'].item(), iters)
                if 'loss_cls' in loss_dict.keys():
                    writer.add_scalar('train/loss_cls', loss_dict['loss_cls'].item(), iters)

            if (i % (len(trainloader) // 10) == 0) and (rank == 0):
                if 'loss_cls' in loss_dict.keys():
                    logger.info('Iters: {:}, Total loss: {:.3f}, loss bbox: {:.3f}, loss giou: {:.3f}, loss cls: {:.3f}'
                                .format(i, total_loss.avg, loss_bbox.avg, loss_giou.avg, loss_cls.avg))
                else:
                    logger.info('Iters: {:}, Total loss: {:.3f}, loss bbox: {:.3f}, loss giou: {:.3f}'
                                .format(i, total_loss.avg, loss_bbox.avg, loss_giou.avg))

        lr_scheduler.step()

        metrics, metrics_name = eval_od(model, valloader, valset.coco,
                                        {'bbox': DINOPostProcess(num_select=cfg.get('num_select', 100),
                                                                 nms_iou_threshold=cfg.get('nms_iou_threshold', -1))
                                        if 'DETRHead' != cfg['model']['decoder']['type'] else DETRPostProcess()})
        # metrics: 'mAP', 'AP@.5', 'AP@.75', 'AP@small', 'AP@medium', 'AP@large'

        if rank == 0:
            eval_results = metric_table(metrics, metrics_name)
            logger.info('Eval Results: \n{}'.format(eval_results))

            writer.add_scalar('eval/mAP', metrics[0], epoch)
            writer.add_scalar('eval/AP@.5', metrics[1], epoch)

        is_best = metrics[0] > previous_best
        previous_best = max(metrics[0], previous_best)
        if rank == 0:
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