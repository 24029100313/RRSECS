import os, cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch
from rsfm.builder.model import ris_model_builder
from rsfm.language.bert.tokenization_bert import BertTokenizer
from rsfm.dataset.transform.transform import resize_fixed



def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
    from scipy.ndimage.morphology import binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        im_overlay[binary_mask] = foreground[binary_mask]

        countours = binary_dilation(binary_mask) ^ binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)


def main():
    image_path = input('please input your image dir: ')
    sentence = input('please describe your prompt: ')

    weights = 'exps/ris/RefCOCO/swin_base_lavt_4xb4_img480_ep40/RefCOCO.swin_base.None.LAVTHead.4xb8.img480.ep40.preimagenet.best73.28.pth'

    dataset_name, backbone, neck, decoder, _, img_size = os.path.basename(weights).split('.')[:6]
    training_size = int(img_size[3:])

    img = Image.open(image_path).convert("RGB")
    img_ndarray = np.array(img)

    src_h, src_w = img.size
    img = resize_fixed(img, mask=None, size=training_size)

    image_transforms = T.Compose(
        [
         T.ToTensor(),
         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    img = image_transforms(img).unsqueeze(0)
    img = img.cuda()

    tokenizer = BertTokenizer.from_pretrained('pretrained_weights/bert/bert-base-uncased')
    sentence_tokenized = tokenizer.encode(text=sentence, add_special_tokens=True)
    sentence_tokenized = sentence_tokenized[:20]
    padded_sent_toks = [0] * 20
    padded_sent_toks[:len(sentence_tokenized)] = sentence_tokenized
    attention_mask = [0] * 20
    attention_mask[:len(sentence_tokenized)] = [1]*len(sentence_tokenized)
    padded_sent_toks = torch.tensor(padded_sent_toks).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
    padded_sent_toks = padded_sent_toks.cuda()
    attention_mask = attention_mask.cuda()

    cfg = {'model': {'backbone': {'type': backbone,
                                  'pretrained': None,
                                  'kwargs': {'in_channels': 3,
                                             'vlf_ris': decoder.replace('Head', '')}},
                     'text_encoder': {'pretrained': 'pretrained_weights/bert/bert-base-uncased'}},
           'dataset': dataset_name,
           'crop_size': training_size,
           'criterion': {'kwargs': {}}}
    if neck != 'None':
        cfg['model']['neck'] = {'type': neck}
    if decoder != 'None':
        cfg['model']['decoder'] = {'type': decoder}

    model = ris_model_builder(cfg)
    model.cuda()
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['model'])

    # inference

    output = model(img, padded_sent_toks, attention_mask)
    output = output.argmax(1, keepdim=True).squeeze().cpu().numpy().astype(np.uint8)
    output = cv2.resize(output, (src_h, src_w), interpolation=cv2.INTER_NEAREST)

    visualization = overlay_davis(img_ndarray, output)
    visualization = Image.fromarray(visualization)
    visualization.show()
    visualization.close()


if __name__ == '__main__':
    main()