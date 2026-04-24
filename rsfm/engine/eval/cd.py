from utils.utils import *
from tqdm import tqdm
import torch.distributed as dist


def eval_cd(model, loader, num_classes, tta=False):
    local_rank = int(os.environ['LOCAL_RANK'])

    model.eval()

    if local_rank == 0:
        tbar = tqdm(total=len(loader), desc='Evaluating:')

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    output_meter = AverageMeter()
    target_meter = AverageMeter()

    with torch.no_grad():
        for img1, img2, mask in loader:
            img1, img2 = img1.cuda(), img2.cuda()

            if tta:
                pred = model(img1, img2, tta).argmax(dim=1)
            else:
                pred = model(img1, img2).argmax(dim=1)

            intersection, union, output, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), num_classes, 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_output = torch.from_numpy(output).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_output)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
            output_meter.update(reduced_output.cpu().numpy())
            target_meter.update(reduced_target.cpu().numpy())

            if local_rank == 0:
                tbar.update(1)

    if local_rank == 0:
        tbar.close()

    OA = np.sum(intersection_meter.sum) / (np.sum(target_meter.sum) + 1e-10) * 100
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIoU = np.mean(iou_class)
    precision = intersection_meter.sum / (output_meter.sum + 1e-10) * 100.0
    recall = intersection_meter.sum / (target_meter.sum + 1e-10) * 100.0
    f1_class = np.array([f_score(x[0], x[1]) for x in zip(precision, recall)])
    mF1 = np.mean(f1_class)

    return [iou_class, f1_class, mIoU, mF1, OA], ['Class', 'IoU', 'F1 Score', 'Overall Accuracy']