import os
import cv2
import torch
import numpy as np
from tqdm import tqdm


def predict_seg(model, loader, args):
    local_rank = int(os.environ['LOCAL_RANK'])

    model.eval()

    if local_rank == 0:
        tbar = tqdm(total=len(loader), desc='Predicting:')

    with torch.no_grad():
        for img, id in loader:
            img = img.cuda()
            if args.tta:
                pred = model(img, args.tta).argmax(dim=1)
            else:
                pred = model(img).argmax(dim=1)
            for i in range(pred.shape[0]):
                output = pred[i].cpu().numpy().astype(np.uint8)
                # cv2.imwrite('%s/%s' % (args.save_path, id[i] + '.png'), output)
                if output.shape != (256, 256):
                    output = cv2.resize(output, (256, 256), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite('%s/%s' % (args.save_path, id[i] + '.tif'), output + 1)

            if local_rank == 0:
                tbar.update(1)

    if local_rank == 0:
        tbar.close()