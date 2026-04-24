import os
import cv2
import torch
import numpy as np
from tqdm import tqdm



def predict_cd(model, loader, args):
    local_rank = int(os.environ['LOCAL_RANK'])

    model.eval()

    if local_rank == 0:
        tbar = tqdm(total=len(loader), desc='Predicting:')

    with torch.no_grad():
        for img1, img2, id in loader:
            img1, img2 = img1.cuda(), img2.cuda()
            if args.tta:
                pred = model(img1, img2, args.tta).argmax(dim=1)
            else:
                pred = model(img1, img2).argmax(dim=1)
            for i in range(pred.shape[0]):
                output = pred[i].cpu().numpy().astype(np.uint8)
                cv2.imwrite('%s/%s' % (args.save_path, id[i] + '.png'), output)

            if local_rank == 0:
                tbar.update(1)

    if local_rank == 0:
        tbar.close()