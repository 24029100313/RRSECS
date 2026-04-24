import os
import numpy as np
from PIL import Image
import torch
from rsfm.builder.model import vg_model_builder
from pytorch_pretrained_bert.tokenization import BertTokenizer
from rsfm.dataset.utils import convert_examples_to_features, read_examples
from rsfm.dataset.transform.transform import make_vg_transforms
from rsfm.utils import VGPostProcess


def overlay_davis(img_path, bbox, colors=[255, 0, 0]):
    import cv2

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    xmin, xmax = int(bbox[0]), int(bbox[2])
    ymin, ymax = int(bbox[1]), int(bbox[3])

    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors, thickness=2)

    return img


def init_input(img_path, expression, img_size):
    infer_transform = make_vg_transforms(img_size, 'val')

    img = Image.open(img_path).convert("RGB")
    bbox = torch.tensor(np.array([0, 0, 0, 0]))
    phrase = expression.lower()

    target = {}
    target['phrase'] = phrase
    target['bbox'] = bbox

    img, target = infer_transform(img, target)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    examples = read_examples(target['phrase'], 0)
    features = convert_examples_to_features(examples=examples, seq_length=20, tokenizer=tokenizer)
    word_id = torch.tensor(features[0].input_ids, dtype=torch.long)
    word_mask = torch.tensor(features[0].input_mask, dtype=torch.bool)

    return img, target['mask'], word_id, word_mask, target['size'], target['ratio'], target['orig_size'], target['dxdy']


def main():
    image_path = input('please input your image dir: ')
    sentence = input('please describe your prompt: ')

    device = 'cuda:7'
    weight_path = 'exps/vg/RefCOCO_VG/swin_tiny_transvg_4xb16_img640_ep50_sa/RefCOCO_VG.swin_tiny.None.TransVGHead.4xb16.img640.ep50.preimagenet.best66.6.pth'
    dataset_name, backbone, neck, decoder, _, img_size = os.path.basename(weight_path).split('.')[:6]
    training_size = int(img_size[3:])

    img, img_mask, word_id, word_mask, size, ratio, orig_size, dxdy = init_input(image_path, sentence, training_size)
    img, img_mask = img.unsqueeze(0).to(device), img_mask.unsqueeze(0).to(device)
    word_id, word_mask = word_id.unsqueeze(0).to(device), word_mask.unsqueeze(0).to(device)
    size, ratio, orig_size, dxdy = size.unsqueeze(0).to(device), ratio.unsqueeze(0).to(device), orig_size.unsqueeze(0).to(device), dxdy.unsqueeze(0).to(device)

    cfg = {'model': {'backbone': {'type': backbone,
                                  'pretrained': None,
                                  'kwargs': {'in_channels': 3}},
                     'text_encoder': {'type': 'bert-base-uncased'}},
           'dataset': dataset_name,
           'crop_size': training_size,
           'criterion': {'kwargs': {'aux_loss': False}}}

    if neck != 'None':
        cfg['model']['neck'] = {'type': neck}
    if decoder != 'None':
        cfg['model']['decoder'] = {'type': decoder,
                                   'kwargs': {'trans_enc': False}}

    model = vg_model_builder(cfg)
    model.to(device)

    ckpt = torch.load(weight_path)['model']
    from collections import OrderedDict
    _tmp = OrderedDict({k.split('.', 1)[1]: v for k, v in ckpt.items()})
    model.load_state_dict(_tmp)

    postprocessor = VGPostProcess()

    # inference
    model.eval()
    with torch.no_grad():
        output = model(img, img_mask, word_id, word_mask)
        output = postprocessor(output, size, ratio, orig_size, dxdy)
        output = output.squeeze(0).cpu().numpy()

    visualization = overlay_davis(image_path, output)
    visualization = Image.fromarray(visualization)
    visualization.show()
    visualization.close()


if __name__ == '__main__':
    main()