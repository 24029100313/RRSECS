import numpy as np
import torch, os, cv2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from rsfm.builder.model import ris_model_builder
from rsfm.dataset.utils import datasets_info
from rsfm.language import BertTokenizer
from torchvision import transforms
from functools import partial
from tqdm import tqdm



def reshape_bnc2bchw(tensor, height, width):
    result = tensor.reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


class model_builder(ris_model_builder):
    def __init__(self, cfg):
        super(model_builder, self).__init__(cfg)

    def forward(self, x):
        l, l_mask = preprocess_text(expression)
        l, l_mask = l.cuda(), l_mask.cuda()

        h, w = x.shape[-2:]
        l_feats, l_mask = self.language_forward(self.text_enc_type, l, l_mask)
        feats = self.backbone(x, l_feats, l_mask)
        if self.neck_cfg:
            feats = self.neck(feats)
        out = self.various_decoder(self.dec_cfg['type'], feats, l_feats, l_mask, h, w)

        return out


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

    model = model_builder(cfg)
    model.cuda()

    checkpoint = torch.load(weight_path)['model']
    from collections import OrderedDict
    _tmp = OrderedDict({k.split('.', 1)[1]: v for k, v in checkpoint.items()})

    # for k in _tmp.keys():
    #     print(k)

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
    tokenizer = BertTokenizer.from_pretrained('pretrained_weights/bert/bert-base-uncased')

    input_id = tokenizer.encode(text=phrase, add_special_tokens=True)
    input_id = input_id[:max_tokens]

    padded_input_id = [0] * max_tokens
    padded_input_id[:len(input_id)] = input_id

    attention_mask = [0] * max_tokens
    attention_mask[:len(input_id)] = [1] * len(input_id)

    input_id = torch.tensor(padded_input_id).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)

    return input_id, attention_mask


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
    rgb_img = np.float32(image) / 255

    model, vision_task, num_classes, dataset_name, img_size = load_model(weight_path)
    model.eval()

    img_tensor = preprocess(img_path, img_size).cuda()
    output = model(img_tensor)
    mask = output.argmax(dim=1).squeeze(0).detach().cpu().numpy()

    mask_float = np.float32(mask == 1)
    targets = [SemanticSegmentationTarget(1, mask_float)]


    depths = [2, 2, 6, 2]
    patch_sizes = [img_size // 2 ** (i + 2)for i in range(len(depths))]

    # # visualize before fusion
    # for i in range(len(depths)):
    #     reshape_transform = partial(reshape_bnc2bchw, height=patch_sizes[i], width=patch_sizes[i])
    #     grad_cam = GradCAM(model=model, target_layers=[model.backbone.layers[i].blocks[depths[i] - 1]],
    #                        reshape_transform=reshape_transform)
    #     cam = grad_cam(input_tensor=img_tensor, targets=targets, aug_smooth=True)[0, :]
    #     cam_image = show_cam_on_image(rgb_img, np.array(cam), use_rgb=True)
    #     vis = Image.fromarray(cam_image)
    #     vis.save(save_dir + '/' + str(img_id) + '_model.backbone.' + 'layers' + str(i) + '.' + 'blocks' + str(depths[i] - 1) + '.png')
    #
    #
    # # visualize after fusion
    # for i in range(len(depths)):
    #     reshape_transform = partial(reshape_bnc2bchw, height=patch_sizes[i], width=patch_sizes[i])
    #     grad_cam = GradCAM(model=model, target_layers=[model.backbone.layers[i].fusion.fusion],
    #                        reshape_transform=reshape_transform)
    #     cam = grad_cam(input_tensor=img_tensor, targets=targets, aug_smooth=True)[0, :]
    #     cam_image = show_cam_on_image(rgb_img, np.array(cam), use_rgb=True)
    #     vis = Image.fromarray(cam_image)
    #     vis.save(save_dir + '/' + str(img_id) + '_model.backbone.' + 'layers' + str(i) + '.fusion.png')
    #
    #
    # # decoder output
    # grad_cam = GradCAM(model=model, target_layers=[model.decoder.conv_seg])
    # cam = grad_cam(input_tensor=img_tensor, targets=targets, aug_smooth=True)[0, :]
    # cam_image = show_cam_on_image(rgb_img, np.array(cam), use_rgb=True)
    # vis = Image.fromarray(cam_image)
    # vis.save(save_dir + '/' + str(img_id) + '_model.decoder.conv_seg.png')

    # multi-scale convolutional module
    for i in range(len(depths)):
        grad_cam = GradCAM(model=model, target_layers=[model.backbone.layers[i].fusion.vis_proj])
        cam = grad_cam(input_tensor=img_tensor, targets=targets, aug_smooth=True)[0, :]
        cam_image = show_cam_on_image(rgb_img, np.array(cam), use_rgb=True)
        vis = Image.fromarray(cam_image)
        vis.save(save_dir + '/' + str(img_id) + '_model.backbone.' + 'layers' + str(i) + '.mcm.png')


    # cv2.imwrite('/root/lxq/18_pred.png', mask * 255)





infos = open('/root/lxq/RSFM_RIS_Datasets/Optical/RefSegRS/phrase_txts/test.txt', 'r').readlines()
weight_path = '/root/Desktop/MCFormer/weights/table_1/swin_t/RefSegRS.swin_tiny.None.MCTHead.4xb4.img512.ep50.preimagenet.best84.42.pth'
for info in tqdm(infos):
    img_id, expression = info.strip().split(' ', 1)
    img_path = '/root/lxq/RSFM_RIS_Datasets/Optical/RefSegRS/images/' + str(img_id) + '.tif'
    save_dir = '/root/Desktop/MCFormer/grad_cam_mcm/' + str(img_id) + ' ' + expression
    plot_grad_cam(img_id, weight_path, img_path, save_dir)
