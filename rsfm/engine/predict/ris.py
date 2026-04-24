import os
import cv2
import torch
import numpy as np
from tqdm import tqdm



def predict_ris(model, loader, args):
    local_rank = int(os.environ['LOCAL_RANK'])

    model.eval()

    if local_rank == 0:
        tbar = tqdm(total=len(loader), desc='Predicting:')

    with torch.no_grad():
        for img, sentence, attention, id in loader:
            img = img.cuda()
            sentence = sentence.cuda().squeeze(1)
            attention = attention.cuda().squeeze(1)

            if args.tta:
                pred = model(img, sentence, attention, args.tta)
            else:
                pred = model(img, sentence, attention)

            if pred.shape[1] > 1:
                pred = pred.argmax(dim=1)
            else:
                pred = torch.sigmoid(pred)
                pred = (pred > 0.5).squeeze(1).int()

            for i in range(pred.shape[0]):
                output = (pred[i].cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite('%s/%s' % (args.save_path, os.path.basename(id[i]).split('.')[0] + '.png'), output)

            if local_rank == 0:
                tbar.update(1)

    if local_rank == 0:
        tbar.close()