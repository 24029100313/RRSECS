from utils.utils import *
from rsfm.dataset import datasets_info
from rsfm.engine.eval import eval_seg



def train_seg(args, cfg, model, optimizer, criterion, trainloader, valloader, trainsampler,
              rank, logger, writer, previous_best, lr_scheduler, epoch):
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best mIoU: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = AverageMeter()

        loss_cls = AverageMeter()
        loss_mask = AverageMeter()
        loss_dice = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):
            img, mask = img.cuda(), mask.cuda()

            pred = model(img)
            loss = criterion(pred, mask)

            torch.distributed.barrier()
            optimizer.zero_grad()

            if isinstance(loss, dict):
                loss['total_loss'].backward()
                optimizer.step()
                lr_scheduler.step()

                total_loss.update(loss['total_loss'].item())
                loss_cls.update(loss['loss_cls'].item())
                loss_mask.update(loss['loss_mask'].item())
                loss_dice.update(loss['loss_dice'].item())

                iters = epoch * len(trainloader) + i
                if rank == 0:
                    writer.add_scalar('train/loss', loss['total_loss'].item(), iters)
                    writer.add_scalar('train/loss_cls', loss['loss_cls'].item(), iters)
                    writer.add_scalar('train/loss_mask', loss['loss_mask'].item(), iters)
                    writer.add_scalar('train/loss_dice', loss['loss_dice'].item(), iters)

                if (i % (len(trainloader) // min(10, len(trainloader))) == 0) and (rank == 0):
                    logger.info('Iters: {:}, Total loss: {:.3f}, Loss cls: {:.3f}, Loss mask: {:.3f}, Loss dice: {:.3f}'
                                .format(i, total_loss.avg, loss_cls.avg, loss_mask.avg, loss_dice.avg))

            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                total_loss.update(loss.item())

                iters = epoch * len(trainloader) + i
                if rank == 0:
                    writer.add_scalar('train/loss', loss.item(), iters)

                if (i % (len(trainloader) // min(10, len(trainloader))) == 0) and (rank == 0):
                    logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        metrics, metrics_name = eval_seg(model, valloader, datasets_info[cfg['dataset']]['num_classes'])
        # metrics: iou_class, f1_class, mIoU, mF1, OA

        if rank == 0:
            eval_results = metric_table(metrics, metrics_name, need_perclass=True,
                                        classes_name=datasets_info[cfg['dataset']]['classes_name'])
            logger.info('Per-class Results: \n{}'.format(eval_results))

            writer.add_scalar('eval/mIoU', metrics[2], epoch)
            writer.add_scalar('eval/mF1', metrics[3], epoch)
            writer.add_scalar('eval/OA', metrics[4], epoch)
            for i, iou in enumerate(metrics[0]):
                writer.add_scalar('eval/%s_IoU' % (datasets_info[cfg['dataset']]['classes_name'][i]), iou, epoch)

        is_best = metrics[2] > previous_best
        previous_best = max(metrics[2], previous_best)
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
                save_best_weight_name = get_save_weight_name(cfg, metrics[2])
                delete_previous_best_weight(args.save_path)
                torch.save(checkpoint, os.path.join(args.save_path, save_best_weight_name))
            if cfg.get('save_interval', -1) > 0:
                if (epoch + 1) % cfg.get('save_interval') == 0:
                    torch.save(checkpoint, os.path.join(args.save_path, 'epoch_{}.pth'.format(epoch + 1)))



# def train_seg(args, cfg, model, optimizer, criterion, trainloader, valloader, trainsampler,
#               rank, logger, writer, previous_best, lr_scheduler, epoch):
#     previous_best_score, previous_best_OA, previous_best_mIoU, previous_best_mF1 = 0., 0., 0., 0.
#
#     for epoch in range(epoch + 1, cfg['epochs']):
#         if rank == 0:
#             logger.info('===========> Epoch: {:}, LR: {:.5f}, '
#                         'Previous best score: {:.2f}, Previous best OA: {:.2f},'
#                         ' Previous best mIoU: {:.2f}, Previous best mF1: {:.2f}'.format(
#                 epoch, optimizer.param_groups[0]['lr'],
#                 previous_best_score, previous_best_OA,
#                 previous_best_mIoU, previous_best_mF1))
#
#         model.train()
#         total_loss = AverageMeter()
#
#         loss_cls = AverageMeter()
#         loss_mask = AverageMeter()
#         loss_dice = AverageMeter()
#
#         trainsampler.set_epoch(epoch)
#
#         for i, (img, mask) in enumerate(trainloader):
#             img, mask = img.cuda(), mask.cuda()
#
#             pred = model(img)
#             loss = criterion(pred, mask)
#
#             torch.distributed.barrier()
#             optimizer.zero_grad()
#
#             if isinstance(loss, dict):
#                 loss['total_loss'].backward()
#                 optimizer.step()
#                 lr_scheduler.step()
#
#                 total_loss.update(loss['total_loss'].item())
#                 loss_cls.update(loss['loss_cls'].item())
#                 loss_mask.update(loss['loss_mask'].item())
#                 loss_dice.update(loss['loss_dice'].item())
#
#                 iters = epoch * len(trainloader) + i
#                 if rank == 0:
#                     writer.add_scalar('train/loss', loss['total_loss'].item(), iters)
#                     writer.add_scalar('train/loss_cls', loss['loss_cls'].item(), iters)
#                     writer.add_scalar('train/loss_mask', loss['loss_mask'].item(), iters)
#                     writer.add_scalar('train/loss_dice', loss['loss_dice'].item(), iters)
#
#                 if (i % (len(trainloader) // 10) == 0) and (rank == 0):
#                     logger.info('Iters: {:}, Total loss: {:.3f}, Loss cls: {:.3f}, Loss mask: {:.3f}, Loss dice: {:.3f}'
#                                 .format(i, total_loss.avg, loss_cls.avg, loss_mask.avg, loss_dice.avg))
#
#             else:
#                 loss.backward()
#                 optimizer.step()
#                 lr_scheduler.step()
#
#                 total_loss.update(loss.item())
#
#                 iters = epoch * len(trainloader) + i
#                 if rank == 0:
#                     writer.add_scalar('train/loss', loss.item(), iters)
#
#                 if (i % (len(trainloader) // 10) == 0) and (rank == 0):
#                     logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))
#
#         metrics, metrics_name = eval_seg(model, valloader, datasets_info[cfg['dataset']]['num_classes'])
#         # metrics: iou_class, f1_class, mIoU, mF1, OA
#
#         if rank == 0:
#             eval_results = metric_table(metrics, metrics_name, need_perclass=True,
#                                         classes_name=datasets_info[cfg['dataset']]['classes_name'])
#             logger.info('Per-class Results: \n{}'.format(eval_results))
#
#             writer.add_scalar('eval/mIoU', metrics[2], epoch)
#             writer.add_scalar('eval/mF1', metrics[3], epoch)
#             writer.add_scalar('eval/OA', metrics[4], epoch)
#             for i, iou in enumerate(metrics[0]):
#                 writer.add_scalar('eval/%s_IoU' % (datasets_info[cfg['dataset']]['classes_name'][i]), iou, epoch)
#
#         best_score = (metrics[4] + metrics[2] + metrics[3]) / 3
#
#         is_best_score = best_score > previous_best_score
#         is_best_OA = metrics[4] > previous_best_OA
#         is_best_mIoU = metrics[2] > previous_best_mIoU
#         is_best_mF1 = metrics[3] > previous_best_mF1
#
#         previous_best_score = max(best_score, previous_best_score)
#         previous_best_OA = max(metrics[4], previous_best_OA)
#         previous_best_mIoU = max(metrics[2], previous_best_mIoU)
#         previous_best_mF1 = max(metrics[3], previous_best_mF1)
#
#
#         if rank == 0:
#             checkpoint = {
#                 'model': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'lr_scheduler': lr_scheduler.state_dict(),
#                 'epoch': epoch,
#                 'previous_best_score': previous_best_score,
#                 'previous_best_OA': previous_best_OA,
#                 'previous_best_mIoU': previous_best_mIoU,
#                 'previous_best_mF1': previous_best_mF1,
#             }
#             torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
#
#             if is_best_score:
#                 save_best_weight_name = get_save_weight_name(cfg, best_score)
#                 delete_previous_best_weight(args.save_path, 'bestscore')
#                 torch.save(checkpoint, os.path.join(args.save_path, save_best_weight_name.replace('best', 'bestscore')))
#
#             if is_best_OA:
#                 save_best_weight_name = get_save_weight_name(cfg, metrics[4])
#                 delete_previous_best_weight(args.save_path, 'bestOA')
#                 torch.save(checkpoint, os.path.join(args.save_path, save_best_weight_name.replace('best', 'bestOA')))
#
#             if is_best_mIoU:
#                 save_best_weight_name = get_save_weight_name(cfg, metrics[2])
#                 delete_previous_best_weight(args.save_path, 'bestmIoU')
#                 torch.save(checkpoint, os.path.join(args.save_path, save_best_weight_name.replace('best', 'bestmIoU')))
#
#             if is_best_mF1:
#                 save_best_weight_name = get_save_weight_name(cfg, metrics[3])
#                 delete_previous_best_weight(args.save_path, 'bestmF1')
#                 torch.save(checkpoint, os.path.join(args.save_path, save_best_weight_name.replace('best', 'bestmF1')))
#
#             if cfg.get('save_interval', -1) > 0:
#                 if (epoch + 1) % cfg.get('save_interval') == 0:
#                     torch.save(checkpoint, os.path.join(args.save_path, 'epoch_{}.pth'.format(epoch + 1)))