from utils.utils import *
from tqdm import tqdm
import torch.distributed as dist


def eval_ris(model, loader, tta=False):
    local_rank = int(os.environ['LOCAL_RANK'])

    model.eval()

    if local_rank == 0:
        tbar = tqdm(total=len(loader), desc='Evaluating:')

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    iou_meter = AverageMeter()
    seg_correct_meter = AverageMeter()
    seg_total_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, sentence, attention in loader:

            img = img.cuda()
            mask = mask.cuda()
            sentence = sentence.cuda().squeeze(1)
            attention = attention.cuda().squeeze(1)

            if tta:
                pred = model(img, sentence, attention, tta)
            else:
                pred = model(img, sentence, attention)

            if pred.shape[1] > 1:
                pred = pred.argmax(dim=1)
            else:
                pred = torch.sigmoid(pred)


            cum_I, cum_U, cum_IoU, seg_correct, seg_total = IoU(pred, mask, eval_seg_iou_list = [.5, .6, .7, .8, .9])
            reduced_seg_correct = torch.from_numpy(seg_correct).cuda()
            reduced_seg_total = torch.from_numpy(np.array(seg_total)).cuda()

            dist.all_reduce(cum_I)
            dist.all_reduce(cum_U)
            dist.all_reduce(cum_IoU)
            dist.all_reduce(reduced_seg_correct)
            dist.all_reduce(reduced_seg_total)

            intersection_meter.update(cum_I.cpu().numpy())
            union_meter.update(cum_U.cpu().numpy())
            iou_meter.update(cum_IoU.cpu().numpy())
            seg_correct_meter.update(reduced_seg_correct.cpu().numpy())
            seg_total_meter.update(reduced_seg_total.cpu().numpy())

            if local_rank == 0:
                tbar.update(1)

    if local_rank == 0:
        tbar.close()

    oIoU = np.sum(intersection_meter.sum) / (np.sum(union_meter.sum) + 1e-10) * 100
    mIoU = np.sum(iou_meter.sum) / np.sum(seg_total_meter.sum) * 100
    seg_correct_avg = list(seg_correct_meter.sum / seg_total_meter.sum * 100)
    seg_correct_avg.append(oIoU)
    seg_correct_avg.append(mIoU)
    seg_correct_avg.append(sum(seg_correct_avg))

    return seg_correct_avg, ['PR@.5', 'PR@.6', 'PR@.7', 'PR@.8', 'PR@.9', 'oIoU', 'mIoU', 'SUM']