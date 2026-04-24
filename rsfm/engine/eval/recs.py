from utils.utils import *
from tqdm import tqdm
import torch.distributed as dist
from rsfm.utils import VGPostProcess
import torch.nn.functional as F


def eval_recs(model, loader, tta=False):
    local_rank = int(os.environ['LOCAL_RANK'])

    postprocessor = VGPostProcess()

    model.eval()

    if local_rank == 0:
        tbar = tqdm(total=len(loader), desc='Evaluating:')

    ris_intersection_meter, vg_intersection_meter = AverageMeter(), AverageMeter()
    ris_union_meter, vg_union_meter = AverageMeter(), AverageMeter()
    ris_iou_meter, vg_iou_meter = AverageMeter(), AverageMeter()
    ris_correct_meter, vg_correct_meter = AverageMeter(), AverageMeter()
    ris_total_meter, vg_total_meter = AverageMeter(), AverageMeter()

    with torch.no_grad():
        for img, img_mask, word_id, word_mask, orig_mask, orig_bbox, size, ratio, orig_size, dxdy, _ in loader:

            img, img_mask = img.cuda(), img_mask.cuda()
            word_id, word_mask = word_id.cuda().squeeze(1), word_mask.cuda().squeeze(1)
            orig_mask, orig_bbox = orig_mask.cuda(), orig_bbox.cuda()
            size, ratio, orig_size, dxdy = size.cuda(), ratio.cuda(), orig_size.cuda(), dxdy.cuda()

            if tta:
                pred = model(img, img_mask, word_id, word_mask, tta)
            else:
                pred = model(img, img_mask, word_id, word_mask)

            pred_ris, pred_vg = pred['mask'], pred['bbox']
            pred_vg = postprocessor(pred_vg, size, ratio, orig_size, dxdy)
            pred_ris = F.interpolate(pred_ris, size=orig_mask.shape[-2:], mode='bilinear', align_corners=False)
            if pred_ris.shape[1] > 1:
                pred_ris = pred_ris.argmax(dim=1)
            else:
                pred_ris = torch.sigmoid(pred_ris)

            # compute metrics for visual grounding
            vg_cum_I, vg_cum_U, vg_cum_IoU, vg_correct, vg_total = \
                BboxIoU(pred_vg, orig_bbox, eval_det_iou_list=[.5, .6, .7, .8, .9])
            reduced_vg_correct = torch.from_numpy(vg_correct).cuda()
            reduced_vg_total = torch.from_numpy(np.array(vg_total)).cuda()

            dist.all_reduce(vg_cum_I)
            dist.all_reduce(vg_cum_U)
            dist.all_reduce(vg_cum_IoU)
            dist.all_reduce(reduced_vg_correct)
            dist.all_reduce(reduced_vg_total)

            vg_intersection_meter.update(vg_cum_I.cpu().numpy())
            vg_union_meter.update(vg_cum_U.cpu().numpy())
            vg_iou_meter.update(vg_cum_IoU.cpu().numpy())
            vg_correct_meter.update(reduced_vg_correct.cpu().numpy())
            vg_total_meter.update(reduced_vg_total.cpu().numpy())

            # compute metrics for referring image segmentation
            ris_cum_I, ris_cum_U, ris_cum_IoU, ris_correct, ris_total = \
                IoU(pred_ris, orig_mask, eval_seg_iou_list=[.5, .6, .7, .8, .9])
            reduced_ris_correct = torch.from_numpy(ris_correct).cuda()
            reduced_ris_total = torch.from_numpy(np.array(ris_total)).cuda()

            dist.all_reduce(ris_cum_I)
            dist.all_reduce(ris_cum_U)
            dist.all_reduce(ris_cum_IoU)
            dist.all_reduce(reduced_ris_correct)
            dist.all_reduce(reduced_ris_total)

            ris_intersection_meter.update(ris_cum_I.cpu().numpy())
            ris_union_meter.update(ris_cum_U.cpu().numpy())
            ris_iou_meter.update(ris_cum_IoU.cpu().numpy())
            ris_correct_meter.update(reduced_ris_correct.cpu().numpy())
            ris_total_meter.update(reduced_ris_total.cpu().numpy())

            if local_rank == 0:
                tbar.update(1)

    if local_rank == 0:
        tbar.close()

    # compute metrics for visual grounding
    vg_oIoU = np.sum(vg_intersection_meter.sum) / (np.sum(vg_union_meter.sum) + 1e-10) * 100
    vg_mIoU = np.sum(vg_iou_meter.sum) / np.sum(vg_total_meter.sum) * 100
    vg_correct_avg = list(vg_correct_meter.sum / vg_total_meter.sum * 100)
    vg_correct_avg.append(vg_oIoU)
    vg_correct_avg.append(vg_mIoU)
    vg_correct_avg.append(sum(vg_correct_avg))

    # compute metrics for referring image segmentation
    ris_oIoU = np.sum(ris_intersection_meter.sum) / (np.sum(ris_union_meter.sum) + 1e-10) * 100
    ris_mIoU = np.sum(ris_iou_meter.sum) / np.sum(ris_total_meter.sum) * 100
    ris_correct_avg = list(ris_correct_meter.sum / ris_total_meter.sum * 100)
    ris_correct_avg.append(ris_oIoU)
    ris_correct_avg.append(ris_mIoU)
    ris_correct_avg.append(sum(ris_correct_avg))

    recs_metrics = ris_correct_avg + vg_correct_avg
    recs_metrics.append(ris_correct_avg[-1] + vg_correct_avg[-1])

    return recs_metrics, ['RIS_P@.5', 'RIS_P@.6', 'RIS_P@.7', 'RIS_P@.8', 'RIS_P@.9', 'RIS_oIoU', 'RIS_mIoU', 'RIS_SUM',
                          'VG_P@.5', 'VG_P@.6', 'VG_P@.7', 'VG_P@.8', 'VG_P@.9', 'VG_oIoU', 'VG_mIoU', 'VG_SUM',
                          'RECS_SUM']