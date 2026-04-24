import torch, os
from torch import nn
import time
from rsfm.dataset.utils import datasets_info
from thop import profile
from rsfm.builder.model import *
from fvcore.nn import FlopCountAnalysis
import yaml
from utils.utils import count_params



def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def calculate_fps(model, img_size=512, l_size=(1, 20), l_mask_size=(1, 20), device="cuda", num_runs=100, need_x_mask=False):
    model = model.to(device)
    model.eval()

    # 生成测试输入
    inputs = torch.randn((1, 3, img_size, img_size)).to(device)
    if need_x_mask:
        x_mask = torch.randint(2, (1, img_size, img_size)).to(device)
    l = torch.randint(0, 10000, l_size).to(device)
    l_mask = torch.randint(0, 2, l_mask_size).to(device)

    # 预热GPU
    for _ in range(10):
        if need_x_mask:
            _ = model(inputs, x_mask, l, l_mask)
        else:
            _ = model(inputs, l, l_mask)

    # 同步CUDA操作
    if device == "cuda":
        torch.cuda.synchronize()

    # 正式计时
    start_time = time.time()
    for _ in range(num_runs):
        if need_x_mask:
            _ = model(inputs, x_mask, l, l_mask)
        else:
            _ = model(inputs, l, l_mask)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    fps = num_runs / elapsed_time
    return fps


def calculate_flops_fvcore(model, img_size=512, l_size=(1, 20), l_mask_size=(1, 20), device="cuda", need_x_mask=True):
    model.eval()
    inputs = torch.randn((1, 3, img_size, img_size)).to(device)
    l = torch.randint(0, 1000, l_size).to(device)
    l_mask = torch.randint(0, 2, l_mask_size).to(device)

    if need_x_mask:
        x_mask = torch.randint(2, (1, img_size, img_size)).to(device)
        flops = FlopCountAnalysis(model, inputs=(inputs, x_mask, l, l_mask))
    else:
        flops = FlopCountAnalysis(model, inputs=(inputs, l, l_mask))

    return flops.total()


def test_model_builder(weight_path):
    dataset_name, backbone, neck, decoder, _, img_size = os.path.basename(weight_path).split('.')[:6]
    training_size = int(img_size[3:])

    vision_task = datasets_info[dataset_name]['vision_task']

    if vision_task in ['referring image segmentation', 'visual grounding', 'recs']:
        text_cfg = {'type': 'bert-base-uncased'}
    else:
        text_cfg = None

    # build config
    cfg = {'model': {'backbone': {'type': backbone,
                                  'pretrained': None,
                                  'kwargs': {'in_channels': 3,
                                             # 'vlf_ris': decoder.replace('Head', '')
                                             'vlf_ris': 'LAVT' if vision_task == 'referring image segmentation' else False
                                             }},
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

    if vision_task == 'semantic segmentation':
        model = seg_model_builder(cfg)

    elif vision_task == 'change detection':
        model = cd_model_builder(cfg)

    elif vision_task == 'scene classification':
        model = cls_model_builder(cfg)

    elif vision_task == 'referring image segmentation':
        model = ris_model_builder(cfg)

    elif vision_task == 'visual grounding':
        model = vg_model_builder(cfg)

    elif vision_task in ['referring expression comprehension and segmentation', 'recs']:
        model = recs_model_builder(cfg)

    else:
        raise NotImplementedError

    print('Loading from {}\n'.format(weight_path))

    model.cuda()
    checkpoint = torch.load(weight_path)['model']
    from collections import OrderedDict
    _tmp = OrderedDict({k.split('.', 1)[1]: v for k, v in checkpoint.items()})
    model.load_state_dict(_tmp)

    return model, training_size





# model, img_size = test_model_builder('/root/data4/RSFM/RefDIOR_SOTAs_Results/RSVG/wosa/weights/swin/RefDIOR_VG.swin_tiny.None.LQVGHead.4xb8.img512.ep50.preimagenet.best81.86.pth')
#
# params = count_parameters(model)
# print(f"Parameters: {params / 1e6:.2f}M")
# fps = calculate_fps(model, img_size, need_x_mask=True)
# print(f"FPS: {fps:.2f}")
# flops = calculate_flops_fvcore(model, img_size, need_x_mask=True)
# print(f"FLOPs: {flops / 1e9:.2f}G")



cfg = yaml.load(open('configs/recs.yaml', "r"), Loader=yaml.Loader)
model = recs_model_builder(cfg).cuda()
print(f"Parameters: {count_params(model):.2f}M")
flops = calculate_flops_fvcore(model, 512)
print(f"FLOPs: {flops / 1e9:.2f}G")



# x = torch.randn((1, 3, 512, 512)).cuda()
# l = torch.randn((1, 768, 20)).cuda()
# l_mask = torch.randn((1, 20, 1)).cuda()
#
# cfg = yaml.load(open('configs/recs.yaml', "r"), Loader=yaml.Loader)
# model = recs_model_builder(cfg).cuda()
# model.eval()
#
# enc = model.backbone
# print(f"Parameters: {count_params(enc):.2f}M")
# flops = FlopCountAnalysis(enc, inputs=(x, l, l_mask)).total()
# print(f"FLOPs: {flops / 1e9:.2f}G")
#
#
# text_enc = model.text_encoder
# print(f"Parameters: {count_params(text_enc):.2f}M")
# flops = FlopCountAnalysis(text_enc,
#                           inputs=(torch.randint(0, 1000, (1, 20)).cuda(),
#                                   torch.randint(0, 2, (1, 20)).cuda())).total()
# print(f"FLOPs: {flops / 1e9:.2f}G")
#
#
# feats = model.backbone(x, l, l_mask)
# dec = model.decoder
# print(f"Parameters: {count_params(dec):.2f}M")
# flops = FlopCountAnalysis(dec,
#                           inputs=(feats, l, l_mask)).total()
# print(f"FLOPs: {flops / 1e9:.2f}G")



# from rsfm.decoder.recs.ccformer.utils import LAGD
#
# class LAGDs(nn.Module):
#     def __init__(self):
#         super(LAGDs, self).__init__()
#
#         self.gate_decoupler = nn.ModuleList([LAGD(256, 768) for _ in range(4)])
#
#     def forward(self, all_features, l, l_mask):
#         ris_feats, vg_feats = [], []
#         for ind, feat in enumerate(all_features):
#             ris_feat, vg_feat = self.gate_decoupler[ind](feat, l, l_mask)
#             ris_feats.append(ris_feat)
#             vg_feats.append(vg_feat)
#
#         return ris_feats, vg_feats
#
# x = [torch.randn((1, 256, 128, 128)).cuda(),
#           torch.randn((1, 256, 64, 64)).cuda(),
#           torch.randn((1, 256, 32, 32)).cuda(),
#           torch.randn((1, 256, 16, 16)).cuda()]
# l = torch.randn((1, 768, 20)).cuda()
# l_mask = torch.randn((1, 20, 1)).cuda()
#
# model = LAGDs().cuda()
# print(f"Parameters: {count_params(model):.2f}M")
# flops = FlopCountAnalysis(model, inputs=(x, l, l_mask)).total()
# print(f"FLOPs: {flops / 1e9:.2f}G")



# from rsfm.module.ris_fusion.ccformer_fusion.ccformer_fusion import CCFormer_fusion
#
# class MCFMs(nn.Module):
#     def __init__(self):
#         super(MCFMs, self).__init__()
#
#         self.MCFMs = nn.ModuleList([CCFormer_fusion(dim, 768) for dim in [96, 192, 384, 768]])
#
#     def forward(self, all_features, l, l_mask):
#         mm_feats = []
#         for ind, feat in enumerate(all_features):
#             vision_f = feat.flatten(2).permute(0, 2, 1)
#             mm_f = self.MCFMs[ind](vision_f, l, l_mask)
#             mm_feats.append(mm_f)
#
#         return mm_feats
#
# x = [torch.randn((1, 96, 128, 128)).cuda(),
#      torch.randn((1, 192, 64, 64)).cuda(),
#      torch.randn((1, 384, 32, 32)).cuda(),
#      torch.randn((1, 768, 16, 16)).cuda()]
# l = torch.randn((1, 768, 20)).cuda()
# l_mask = torch.randn((1, 20, 1)).cuda()
#
# model = MCFMs().cuda()
# print(f"Parameters: {count_params(model):.2f}M")
# flops = FlopCountAnalysis(model, inputs=(x, l, l_mask)).total()
# print(f"FLOPs: {flops / 1e9:.2f}G")