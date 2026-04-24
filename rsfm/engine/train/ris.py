from utils.utils import *
from rsfm.engine.eval import eval_ris



def train_ris(args, cfg, model, optimizer, criterion, trainloader, valloader, trainsampler,
              rank, logger, writer, previous_best, lr_scheduler, epoch):
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best oIoU: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = AverageMeter()

        loss_dice = AverageMeter()
        loss_focal = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask, sentence, attention) in enumerate(trainloader):
            img, mask = img.cuda(), mask.cuda()
            sentence, attention = sentence.cuda().squeeze(1), attention.cuda().squeeze(1)

            pred = model(img, sentence, attention)
            loss = criterion(pred, mask)

            optimizer.zero_grad()

            if isinstance(loss, dict):
                loss['total_loss'].backward()
                optimizer.step()
                lr_scheduler.step()

                total_loss.update(loss['total_loss'].item())
                loss_dice.update(loss['loss_dice'].item())
                loss_focal.update(loss['loss_focal'].item())

                iters = epoch * len(trainloader) + i
                if rank == 0:
                    writer.add_scalar('train/loss', loss['total_loss'].item(), iters)
                    writer.add_scalar('train/loss_dice', loss['loss_dice'].item(), iters)
                    writer.add_scalar('train/loss_focal', loss['loss_focal'].item(), iters)

                if (i % (max(len(trainloader) // 10, 2)) == 0) and (rank == 0):
                    logger.info('Iters: {:}, Total loss: {:.3f}, Loss dice: {:.3f}, Loss focal: {:.3f}'
                                .format(i, total_loss.avg, loss_dice.avg, loss_focal.avg))
            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                total_loss.update(loss.item())
                iters = epoch * len(trainloader) + i

                if rank == 0:
                    writer.add_scalar('train/loss', loss.item(), iters)

                if (i % (max(len(trainloader) // 10, 2)) == 0) and (rank == 0):
                    logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        metrics, metrics_name = eval_ris(model, valloader)
        # metrics: PR@.5, PR@.6, PR@.7, PR@.8, PR@.9, oIoU, mIoU, SUM

        if rank == 0:
            eval_results = metric_table(metrics, metrics_name)
            logger.info('Eval Results: \n{}'.format(eval_results))
            for metric, metric_name in zip(metrics, metrics_name):
                writer.add_scalar('eval/' + metric_name, metric, epoch)

        is_best = metrics[-1] > previous_best
        previous_best = max(metrics[-1], previous_best)
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
                save_best_weight_name = 'best_sum{}_epoch{}.pth'.format(round(metrics[-1], 2), epoch + 1)
                delete_previous_best_weight(args.save_path, 'best')
                torch.save({'model': model.state_dict()}, os.path.join(args.save_path, save_best_weight_name))
            if cfg.get('save_interval', -1) > 0:
                if (epoch + 1) % cfg.get('save_interval') == 0:
                    torch.save(checkpoint, os.path.join(args.save_path, 'epoch_{}.pth'.format(epoch + 1)))
