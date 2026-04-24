import os
import torch
import numpy as np
from tqdm import tqdm



def predict_cls(model, loader, args):
    local_rank = int(os.environ['LOCAL_RANK'])

    model.eval()

    if local_rank == 0:
        tbar = tqdm(total=len(loader), desc='Predicting:')

    class_to_idx = loader.classes_dict
    idx_to_class = dict((v, k) for k, v in class_to_idx.items())
    with torch.no_grad():
        for img, id in loader:
            img = img.cuda()
            if args.tta:
                pred = model(img, args.tta).argmax(dim=1)
            else:
                pred = model(img).argmax(dim=1)
            for i in range(pred.shape[0]):
                output = pred[i].cpu().numpy().astype(np.uint8)
                print('The class of image {} is {}'.format(id[i], idx_to_class[output]))

            if local_rank == 0:
                tbar.update(1)

    if local_rank == 0:
        tbar.close()