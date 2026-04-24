import numpy as np
import cv2
import mmcv
import os


num_classes = 9
reduce_zero = True
img_size = 256
proposals = ['/root/data4/RSFM/RSFMv0.3.2/exps/mmseg/MMSeg_YREB/432_mmnorm/swin_tiny_upernet_2xb8_ep50_img512_fold1/test/MMSeg_YREB.swin_tiny.None.UPerHead.2xb8.img512.ep50.preimagenet.bestscore62.13_predictions',
             '/root/data4/RSFM/RSFMv0.3.2/exps/mmseg/MMSeg_YREB/432_mmnorm/swin_tiny_duat256_2xb8_ep50_img512_fold1/test/MMSeg_YREB.swin_tiny.None.DuATHead.2xb8.img512.ep50.preimagenet.bestscore62.14_predictions',
             '/root/data4/RSFM/RSFMv0.3.2/exps/mmseg/MMSeg_YREB/432_mmnorm/swin_tiny_fpn_2xb8_ep50_img512_fold1/test/MMSeg_YREB.swin_tiny.None.FPNHead.2xb8.img512.ep50.preimagenet.bestscore61.51_predictions',
             '/root/data4/RSFM/RSFMv0.3.2/exps/mmseg/MMSeg_YREB/432_mmnorm/swin_tiny_emcad_2xb8_ep50_img512_fold1/test/MMSeg_YREB.swin_tiny.None.EMCADHead.2xb8.img512.ep50.preimagenet.bestscore61.03_predictions',
             '/root/data4/RSFM/RSFMv0.3.2/exps/mmseg/MMSeg_YREB/432_mmnorm/swin_tiny_unetformer256_2xb8_ep50_img512_fold1/test/MMSeg_YREB.swin_tiny.None.UNetFormerHead.2xb8.img512.ep50.preimagenet.bestscore60.58_predictions',
             '/root/data4/RSFM/RSFMv0.3.2/exps/mmseg/MMSeg_YREB/432_mmnorm/swin_tiny_segformer512_2xb8_ep50_img512_fold1/test/MMSeg_YREB.swin_tiny.None.SegFormerHead.2xb8.img512.ep50.preimagenet.bestscore62.13_predictions',
             '/root/data4/RSFM/RSFMv0.3.2/exps/mmseg/MMSeg_YREB/432_mmnorm/swin_tiny_lightham_2xb8_ep50_img512_fold1/test/MMSeg_YREB.swin_tiny.None.LightHamHead.2xb8.img512.ep50.preimagenet.bestscore62.4_predictions',
             '/root/data4/RSFM/RSFMv0.3.2/exps/mmseg/MMSeg_YREB/432_mmnorm/swin_tiny_unet_2xb8_ep50_img512_fold1/test/MMSeg_YREB.swin_tiny.None.UNetHead.2xb8.img512.ep50.preimagenet.bestscore60.5_predictions',
             '/root/data4/RSFM/RSFMv0.3.2/exps/mmseg/MMSeg_YREB/432_mmnorm/swin_tiny_unetv2256_2xb8_ep50_img512_fold1/test/MMSeg_YREB.swin_tiny.None.UNetv2Head.2xb8.img512.ep50.preimagenet.bestscore62.36_predictions',
             '/root/data4/RSFM/RSFMv0.3.2/exps/mmseg/MMSeg_YREB/432_mmnorm/swin_tiny_aerialformer_2xb8_ep50_img512_fold1/test/MMSeg_YREB.swin_tiny.None.AerialFormerHead.2xb8.img512.ep50.preimagenet.bestscore58.07_predictions',
             '/root/data4/RSFM/RSFMv0.3.2/exps/mmseg/MMSeg_YREB/432_mmnorm/swin_tiny_banet_2xb8_ep50_img512_fold1/test/MMSeg_YREB.swin_tiny.None.BANetHead.2xb8.img512.ep50.preimagenet.bestscore61.38_predictions',
             '/root/data4/RSFM/RSFMv0.3.2/exps/mmseg/MMSeg_YREB/432_mmnorm/swin_tiny_umixformer_2xb8_ep50_img512_fold1/test/MMSeg_YREB.swin_tiny.None.UMixFormerHead.2xb8.img512.ep50.preimagenet.bestscore62.1_predictions']
names = os.listdir(proposals[0])
save_path = '/root/data4/RSFM/RSFMv0.3.2/exps/mmseg/MMSeg_YREB/Test_vote/1/results'
os.makedirs(save_path, exist_ok=True)
wr = open(save_path.replace('results', 'vote_proposals.txt'), 'a')
for p in proposals:
    wr.write(p + '\n')


def hard_vote(name):
    final_result = np.zeros((num_classes, img_size, img_size), dtype=np.uint8)
    for proposal in proposals:
        img = cv2.imread(proposal + '/' + name, 0)
        for i in range(num_classes):
            cls = i
            if reduce_zero:
                cls = i + 1
            final_result[i][img == cls] += 1
    output = np.argmax(final_result, 0).astype(np.uint8)
    if reduce_zero:
        output += 1
    cv2.imwrite(save_path + '/' + name, output)

mmcv.track_parallel_progress(hard_vote, names, 64)