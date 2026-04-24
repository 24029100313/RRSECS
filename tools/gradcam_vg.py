import numpy as np
import torch, os, cv2
from PIL import Image
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from rsfm.dataset.utils.utils import datasets_info, convert_examples_to_features, read_examples
from rsfm.builder.model import vg_model_builder
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torchvision import transforms
from rsfm.utils import NestedTensor
from tqdm import tqdm
from functools import partial


def reshape_bnc2bchw(tensor, height, width):
    result = tensor.reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


class model_builder(vg_model_builder):
    def __init__(self, cfg, size):
        super(model_builder, self).__init__(cfg)

        self.size = size

    def forward(self, x):
        x_mask = torch.zeros((self.size, self.size), dtype=torch.bool)
        x_mask = x_mask.unsqueeze(0).cuda()

        l, l_mask = preprocess_text(expression)
        l, l_mask = l.unsqueeze(0).cuda(), l_mask.unsqueeze(0).cuda()
        text_feat, text_mask = self.language_forward(self.text_enc_type, l, l_mask)

        if self.dec_cfg.get('type') == 'QRNetHead':
            feats = self.backbone(x, text_feat[:, 0])
        elif self.dec_cfg.get('type') == 'LPVAHead':
            feats = self.backbone(x, text_feat)
        else:
            feats = self.backbone(x)

        if self.neck_cfg:
            if self.neck_cfg.get('type') == 'QMF':
                feats = self.neck(feats, text_feat[:, 0])
            else:
                feats = self.neck(feats)

        outs, pos = [], []
        for feat in feats:
            mask = F.interpolate(x_mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out = NestedTensor(feat, mask)
            outs.append(out)
            pos.append(self.pos_embed(out).to(out.tensors.dtype))

        res = self.decoder(outs, pos, text_feat, text_mask)

        return res


def load_model(weight_path):
    dataset_name, backbone, neck, decoder, _, img_size = os.path.basename(weight_path).split('.')[:6]
    training_size = int(img_size[3:])

    vision_task = datasets_info[dataset_name]['vision_task']
    num_classes = datasets_info[dataset_name]['num_classes']

    if vision_task in ['referring image segmentation', 'visual grounding']:
        text_cfg = {'type': 'bert-base-uncased'}
    else:
        text_cfg = None

    # build config
    cfg = {'model': {'backbone': {'type': backbone,
                                  'pretrained': None,
                                  'kwargs': {'in_channels': 3,
                                             'vlf_ris': decoder.replace('Head', '')
                                             if vision_task == 'referring image segmentation' else False}},
                     'text_encoder': text_cfg
                     },
           'dataset': dataset_name,
           'crop_size': training_size,
           'criterion': {'kwargs': {}}}
    if neck != 'None':
        cfg['model']['neck'] = {'type': neck}
    if decoder != 'None':
        cfg['model']['decoder'] = {'type': decoder,
                                   'kwargs': {'trans_enc': False}}

    model = model_builder(cfg, training_size)
    model.cuda()

    checkpoint = torch.load(weight_path)['model']
    from collections import OrderedDict
    _tmp = OrderedDict({k.split('.', 1)[1]: v for k, v in checkpoint.items()})

    model.load_state_dict(_tmp)

    return model, vision_task, num_classes, dataset_name, training_size


def preprocess(img_path, size):
    processes = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = processes(img).unsqueeze(0)

    return img_tensor


def preprocess_text(phrase):
    max_tokens = 20
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    examples = read_examples(phrase, 1)
    features = convert_examples_to_features(examples=examples, seq_length=max_tokens, tokenizer=tokenizer)
    word_id = torch.tensor(features[0].input_ids, dtype=torch.long)
    word_mask = torch.tensor(features[0].input_mask, dtype=torch.bool)

    return word_id, word_mask


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


def plot_grad_cam(img_id, weight_path, img_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    image = np.array(Image.open(img_path))

    model, vision_task, num_classes, dataset_name, img_size = load_model(weight_path)
    model.eval()

    image = cv2.resize(image, (img_size, img_size))
    rgb_img = np.float32(image) / 255

    img_tensor = preprocess(img_path, img_size).cuda()

    depths = [2, 2, 6, 2]
    patch_sizes = [img_size // 2 ** (i + 2) for i in range(len(depths))]


    # visualize before fusion
    for i in range(len(depths)):
        reshape_transform = partial(reshape_bnc2bchw, height=patch_sizes[i], width=patch_sizes[i])
        grad_cam = GradCAM(model=model, target_layers=[model.backbone.layers[i].blocks[depths[i] - 1]],
                           reshape_transform=reshape_transform)
        cam = grad_cam(img_tensor)[0, :]
        cam_image = show_cam_on_image(rgb_img, np.array(cam), use_rgb=True)
        vis = Image.fromarray(cam_image)
        vis.save(save_dir + '/' + str(img_id) + '_model.backbone.' + 'layers' + str(i) + '.' + 'blocks' + str(depths[i] - 1) + '.png')


infos = open('/root/lxq/RSFM_VG_Datasets/Optical/DIOR-RSVG/phrase_txts/test.txt', 'r').readlines()
weight_path = '/root/lxq/DIOR-RSVG.swin_tiny.None.TransVGHead.4xb4.img512.ep50.preimagenet.best71.08.pth'
for info in tqdm(infos):
    need = info.strip().split(' ')
    img_id = need[0]
    expression = ' '.join(i for i in need[5:])
    img_path = '/root/lxq/RSFM_VG_Datasets/Optical/DIOR-RSVG/images/' + str(img_id) + '.jpg'
    save_dir = '/root/Desktop/VG/'
    plot_grad_cam(img_id, weight_path, img_path, save_dir)
