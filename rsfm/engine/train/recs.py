from utils.utils import *
from rsfm.engine.eval import eval_recs



def train_recs(args, cfg, model, optimizer, criterion, trainloader, valloader, trainsampler,
               rank, logger, writer, previous_best, lr_scheduler, epoch):
    previous_best_ris = previous_best_vg = previous_best_recs = previous_best

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.7f}, Previous Best RIS: {:.2f}, '
                        'Previous Best VG: {:.2f}, Previous Best RECS: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best_ris, previous_best_vg, previous_best_recs))

        model.train()
        total_loss = AverageMeter()
        loss_ris = AverageMeter()
        loss_vg = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, img_mask, word_id, word_mask, mask, bbox) in enumerate(trainloader):
            img, img_mask = img.cuda(), img_mask.cuda()
            word_id, word_mask = word_id.cuda().squeeze(1), word_mask.cuda().squeeze(1)
            mask, bbox = mask.cuda(), bbox.cuda()

            pred = model(img, img_mask, word_id, word_mask)
            loss_dict = criterion(pred, mask, bbox)

            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss.update(loss_dict['total_loss'].item())
            loss_ris.update(loss_dict['loss_ris'].item())
            loss_vg.update(loss_dict['loss_vg'].item())

            iters = epoch * len(trainloader) + i
            if rank == 0:
                writer.add_scalar('train/loss', loss_dict['total_loss'].item(), iters)
                writer.add_scalar('train/loss_ris', loss_dict['loss_ris'].item(), iters)
                writer.add_scalar('train/loss_vg', loss_dict['loss_vg'].item(), iters)

                # logger.info('Iters: {:}, Total loss: {:.3f}, loss ris: {:.3f}, loss vg: {:.3f}'
                #             .format(i, total_loss.avg, loss_ris.avg, loss_vg.avg))

            if (i % (len(trainloader) // 5) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, loss ris: {:.3f}, loss vg: {:.3f}'
                            .format(i, total_loss.avg, loss_ris.avg, loss_vg.avg))

        metrics, metrics_name = eval_recs(model, valloader)
        # RIS_P@.5, RIS_P@.6, RIS_P@.7, RIS_P@.8, RIS_P@.9, RIS_oIoU, RIS_mIoU, RIS_SUM,
        # VG_P@.5, VG_P@.6, VG_P@.7, VG_P@.8, VG_P@.9, VG_oIoU, VG_mIoU, VG_SUM, RECS_SUM

        if rank == 0:
            eval_results = metric_table(metrics, metrics_name)
            logger.info('Eval Results: \n{}'.format(eval_results))
            for metric, metric_name in zip(metrics, metrics_name):
                writer.add_scalar('eval/' + metric_name, metric, epoch)

        is_best_ris = metrics[7] > previous_best_ris
        is_best_vg = metrics[-2] > previous_best_vg
        is_best_recs = metrics[-1] > previous_best_recs

        previous_best_ris = max(metrics[7], previous_best_ris)
        previous_best_vg = max(metrics[-2], previous_best_vg)
        previous_best_recs = max(metrics[-1], previous_best_recs)

        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'previous_best_ris': previous_best_ris,
                'previous_best_vg': previous_best_vg,
                'previous_best_recs': previous_best_recs,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))

            model_ckpt = {'model': model.state_dict()}

            if is_best_ris:
                save_best_weight_name = 'best_ris_sum{}_epoch{}.pth'.format(round(metrics[7], 2), epoch + 1)
                delete_previous_best_weight(args.save_path, 'best_ris')
                torch.save(model_ckpt, os.path.join(args.save_path, save_best_weight_name))

            if is_best_vg:
                save_best_weight_name = 'best_vg_sum{}_epoch{}.pth'.format(round(metrics[-2], 2), epoch + 1)
                delete_previous_best_weight(args.save_path, 'best_vg')
                torch.save(model_ckpt, os.path.join(args.save_path, save_best_weight_name))

            if is_best_recs:
                save_best_weight_name = 'best_recs_sum{}_epoch{}.pth'.format(round(metrics[-1], 2), epoch + 1)
                delete_previous_best_weight(args.save_path, 'best_recs')
                torch.save(model_ckpt, os.path.join(args.save_path, save_best_weight_name))

            if cfg.get('save_interval', -1) > 0:
                if (epoch + 1) % cfg.get('save_interval') == 0:
                    torch.save(model_ckpt, os.path.join(args.save_path, 'epoch_{}.pth'.format(epoch + 1)))