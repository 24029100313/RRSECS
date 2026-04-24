import torch.distributed as dist
from utils.utils import *
from tqdm import tqdm
from rsfm.utils import VGPostProcess
import torch.nn.functional as F
import cv2


def predict_recs(model, loader, args):
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()

    postprocessor = VGPostProcess()

    model.eval()

    save_dict = {}
    if local_rank == 0:
        tbar = tqdm(total=len(loader), desc='Predicting:')

    with torch.no_grad():
        for img, img_mask, word_id, word_mask, _, _, size, ratio, orig_size, dxdy, uni_id in loader:

            img, img_mask = img.cuda(), img_mask.cuda()
            word_id, word_mask = word_id.cuda().squeeze(1), word_mask.cuda().squeeze(1)
            size, ratio, orig_size, dxdy = size.cuda(), ratio.cuda(), orig_size.cuda(), dxdy.cuda()

            if args.tta:
                pred = model(img, img_mask, word_id, word_mask, args.tta)
            else:
                pred = model(img, img_mask, word_id, word_mask)
            
            pred_ris, pred_vg = pred['mask'], pred['bbox']
            pred_vg = postprocessor(pred_vg, size, ratio, orig_size, dxdy)
            src_h, src_w = orig_size.unbind(1)
            pred_ris = F.interpolate(pred_ris, size=(int(src_h.item()), int(src_w.item())), mode='bilinear', align_corners=False)
            if pred_ris.shape[1] > 1:
                pred_ris = pred_ris.argmax(dim=1)
            else:
                pred_ris = torch.sigmoid(pred_ris)
                pred_ris = (pred_ris > 0.5).squeeze(1).int()

            for i in range(pred_ris.shape[0]):
                output = (pred_ris[i].cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite('%s/%s' % (args.save_ris_path, uni_id[i] + '.png'), output)
                save_dict[uni_id[i]] = [round(v, 3) for v in pred_vg[i].cpu().numpy().tolist()]

                # cv2.imwrite('%s/%s' % (args.save_ris_path, str(uni_id[i].item()) + '.png'), output)
                # save_dict[str(uni_id[i].item())] = [round(v, 3) for v in pred_vg[i].cpu().numpy().tolist()]

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

        with open(os.path.join(args.save_path, 'predictions.json'), 'w') as f:
            json.dump(final_results, f)