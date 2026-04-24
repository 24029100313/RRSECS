from utils.utils import *
from tqdm import tqdm
from utils.coco_eval import CocoEvaluator


def eval_od(model, loader, base_ds, postprocessors, tta=False):
    local_rank = int(os.environ['LOCAL_RANK'])

    model.eval()

    coco_evaluator = CocoEvaluator(base_ds, ('bbox', ), useCats=True)

    if local_rank == 0:
        tbar = tqdm(total=len(loader), desc='Evaluating:')

    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.to(torch.device('cuda'))
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            if tta:
                outputs = model(samples, tta)
            else:
                outputs = model(samples)

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)

            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            coco_evaluator.update(res)

            if local_rank == 0:
                tbar.update(1)

    if local_rank == 0:
        tbar.close()

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return [round(100 * i, 2) for i in coco_evaluator.coco_eval['bbox'].stats.tolist()[:6]], \
           ['mAP', 'AP@.5', 'AP@.75', 'AP@small', 'AP@medium', 'AP@large']