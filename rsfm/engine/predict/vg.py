import json, os
import torch
from tqdm import tqdm
import torch.distributed as dist
from rsfm.utils import VGPostProcess



def predict_vg(model, loader, args):
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()

    postprocessor = VGPostProcess()

    model.eval()

    save_dict = {}
    if local_rank == 0:
        tbar = tqdm(total=len(loader), desc='Predicting:')

    with torch.no_grad():
        for img, img_mask, word_id, word_mask, _, size, ratio, orig_size, dxdy, uni_id in loader:
            img, img_mask = img.cuda(), img_mask.cuda()
            word_id, word_mask = word_id.cuda(), word_mask.cuda()
            size, ratio, orig_size, dxdy = size.cuda(), ratio.cuda(), orig_size.cuda(), dxdy.cuda()

            if args.tta:
                pred = model(img, img_mask, word_id, word_mask, args.tta)
            else:
                pred = model(img, img_mask, word_id, word_mask)

            pred = postprocessor(pred, size, ratio, orig_size, dxdy)

            for i in range(pred.shape[0]):
                save_dict[uni_id[i]] = [round(v, 3) for v in pred[i].cpu().numpy().tolist()]

            if local_rank == 0:
                tbar.update(1)

    if local_rank == 0:
        tbar.close()

    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(all_results, save_dict)

    if local_rank == 0:
        final_results = {}
        for result in all_results:
            final_results.update(result)

        with open(os.path.join(args.save_path, 'predictions_val.json'), 'w') as f:
            json.dump(final_results, f)