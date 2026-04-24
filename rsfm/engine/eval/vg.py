from utils.utils import *
from tqdm import tqdm
import torch.distributed as dist
from rsfm.utils import VGPostProcess


def eval_vg(model, loader, tta=False):
    local_rank = int(os.environ['LOCAL_RANK'])

    postprocessor = VGPostProcess()

    model.eval()

    if local_rank == 0:
        tbar = tqdm(total=len(loader), desc='Evaluating:')

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    iou_meter = AverageMeter()
    det_correct_meter = AverageMeter()
    det_total_meter = AverageMeter()

    with torch.no_grad():
        for img, img_mask, word_id, word_mask, orig_bbox, size, ratio, orig_size, dxdy, _ in loader:

            img, img_mask = img.cuda(), img_mask.cuda()
            word_id, word_mask = word_id.cuda(), word_mask.cuda()
            orig_bbox = orig_bbox.cuda()
            size, ratio, orig_size, dxdy = size.cuda(), ratio.cuda(), orig_size.cuda(), dxdy.cuda()

            if tta:
                pred = model(img, img_mask, word_id, word_mask, tta)
            else:
                pred = model(img, img_mask, word_id, word_mask)

            pred = postprocessor(pred, size, ratio, orig_size, dxdy)

            cum_I, cum_U, cum_IoU, det_correct, det_total = BboxIoU(pred, orig_bbox, [.5, .6, .7, .8, .9])
            reduced_det_correct = torch.from_numpy(det_correct).cuda()
            reduced_det_total = torch.from_numpy(np.array(det_total)).cuda()

            dist.all_reduce(cum_I)
            dist.all_reduce(cum_U)
            dist.all_reduce(cum_IoU)
            dist.all_reduce(reduced_det_correct)
            dist.all_reduce(reduced_det_total)

            intersection_meter.update(cum_I.cpu().numpy())
            union_meter.update(cum_U.cpu().numpy())
            iou_meter.update(cum_IoU.cpu().numpy())
            det_correct_meter.update(reduced_det_correct.cpu().numpy())
            det_total_meter.update(reduced_det_total.cpu().numpy())

            if local_rank == 0:
                tbar.update(1)

    if local_rank == 0:
        tbar.close()

    oIoU = np.sum(intersection_meter.sum) / (np.sum(union_meter.sum) + 1e-10) * 100
    mIoU = np.sum(iou_meter.sum) / np.sum(det_total_meter.sum) * 100
    det_correct_avg = list(det_correct_meter.sum / det_total_meter.sum * 100)
    det_correct_avg.append(oIoU)
    det_correct_avg.append(mIoU)
    det_correct_avg.append(sum(det_correct_avg))

    return det_correct_avg, ['PR@.5', 'PR@.6', 'PR@.7', 'PR@.8', 'PR@.9', 'oIoU', 'mIoU', 'SUM']