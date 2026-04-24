from utils.utils import *
from rsfm.dataset import datasets_info
from rsfm.engine.eval import eval_cd



def train_cd(args, cfg, model, optimizer, criterion, trainloader, valloader, trainsampler,
             rank, logger, writer, previous_best, lr_scheduler, epoch):
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best mIoU: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img1, img2, mask) in enumerate(trainloader):
            img1, img2, mask = img1.cuda(), img2.cuda(), mask.cuda()

            pred = model(img1, img2)
            loss = criterion(pred, mask)

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)

            if (i % (len(trainloader) // 10) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        metrics, metrics_name = eval_cd(model, valloader, datasets_info[cfg['dataset']]['num_classes'])
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