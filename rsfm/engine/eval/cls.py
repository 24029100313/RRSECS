from utils.utils import *
from tqdm import tqdm
import torch.distributed as dist


# def eval_cls(model, loader, num_classes, tta=False):
#     local_rank = int(os.environ['LOCAL_RANK'])
#
#     model.eval()
#
#     if local_rank == 0:
#         tbar = tqdm(total=len(loader), desc='Evaluating:')
#
#     accuracy_1_meter = AverageMeter()
#     accuracy_5_meter = AverageMeter()
#     output_meter = AverageMeter()
#     target_meter = AverageMeter()
#
#     with torch.no_grad():
#         for img, label in loader:
#             img = img.cuda()
#
#             if tta:
#                 output = model(img, tta)
#             else:
#                 output = model(img)
#
#             _, pred = torch.topk(output, k=5, dim=1, largest=True, sorted=True)
#
#             accuracy_1, accuracy_5, output, target = \
#                 AccuracyCalculate(pred.cpu().numpy(), label.unsqueeze(dim=-1).numpy(), num_classes)
#
#             reduced_accuracy_1 = torch.from_numpy(accuracy_1).cuda()
#             reduced_accuracy_5 = torch.from_numpy(accuracy_5).cuda()
#             reduced_output = torch.from_numpy(output).cuda()
#             reduced_target = torch.from_numpy(target).cuda()
#
#             dist.all_reduce(reduced_accuracy_1)
#             dist.all_reduce(reduced_accuracy_5)
#             dist.all_reduce(reduced_output)
#             dist.all_reduce(reduced_target)
#
#             accuracy_1_meter.update(reduced_accuracy_1.cpu().numpy())
#             accuracy_5_meter.update(reduced_accuracy_5.cpu().numpy())
#             output_meter.update(reduced_output.cpu().numpy())
#             target_meter.update(reduced_target.cpu().numpy())
#
#             if local_rank == 0:
#                 tbar.update(1)
#
#     if local_rank == 0:
#         tbar.close()
#
#     Acc1_class = accuracy_1_meter.sum / (target_meter.sum + 1e-10) * 100.0
#     mAcc1 = np.mean(Acc1_class)
#     Acc5_class = accuracy_5_meter.sum / (target_meter.sum + 1e-10) * 100.0
#     mAcc5 = np.mean(Acc5_class)
#
#     return [mAcc1, mAcc5], ['Top1 mAcc', 'Top5 mAcc']


def eval_cls(model, loader, num_classes, tta=False):
    local_rank = int(os.environ['LOCAL_RANK'])

    model.eval()

    if local_rank == 0:
        tbar = tqdm(total=len(loader), desc='Evaluating:')

    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    with torch.no_grad():
        for img, label in loader:
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=True):
                if tta:
                    output = model(img, tta)
                else:
                    output = model(img)

            acc1, acc5 = accuracy(output, label, topk=(1, 5))

            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)

            acc1_meter.update(acc1.item(), label.size(0))
            acc5_meter.update(acc5.item(), label.size(0))

            if local_rank == 0:
                tbar.update(1)

    if local_rank == 0:
        tbar.close()

    return [acc1_meter.avg, acc5_meter.avg], ['Top1 mAcc', 'Top5 mAcc']