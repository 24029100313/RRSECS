# Remote Sensing Foundation Model --- IPIU-XDU

## News
- `2024/07/09`: Support vision-language downstream task now! Add [LAVT](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_LAVT_Language-Aware_Vision_Transformer_for_Referring_Image_Segmentation_CVPR_2022_paper.html), [RRSIS](https://ieeexplore.ieee.org/abstract/document/10458079), and [RMSIN](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Rotated_Multi-Scale_Interaction_Network_for_Referring_Remote_Sensing_Image_Segmentation_CVPR_2024_paper.html) for referring image segmentation.
- `2024/07/09`: Contributions to the build of scene classification: data processor and running engine by [Qin Ma](https://github.com/chunbai1) and [Xinyu Liu](https://github.com/xxxxyliu), MiT and PVTv2 by [Jing Zhang](https://github.com/Jerry-jing), LSKNet and ViT by [Jiamin Cao](https://github.com/JMcarrot), Swin Transformer and ConvNeXt by [Chenyue Che](https://github.com/chenyueche), FocalNet by [Yanyan Zu](https://github.com/Zuyanyan), InternImage by [Yanzhao Zhang](https://github.com/stuzyz), VMamba and UniRepLKNet by [Jinming Chai](https://github.com/JMcarrot).
- `2024/07/09`: Release v0.2.0. Unify codes for semantic segmentation, change detection, scene classification, and referring image segmentation.
- `2024/07/01`: Add [ChangeMamba](https://ieeexplore.ieee.org/document/10565926), [BIT_CD](https://ieeexplore.ieee.org/document/9491802), and [UNet](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) in `model/decoder`.
- `2024/06/24`: Add [SpatSIGMA](https://arxiv.org/abs/2406.11519) and [HyperSIGMA](https://arxiv.org/abs/2406.11519) in `model/backbone/hypersigma`, which the latter needs to pass a list to `cfg['model']['backbone']['pretrained']`.
- `2024/06/23`: Add [SpectralGPT](https://arxiv.org/abs/2311.07113) in `model/backbone/spectralgpt`, by [Xinyu Liu](https://github.com/xxxxyliu).
- `2024/06/23`: Add [Swin Transformer v2](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html) family models in `model/backbone/swin_transformer_v2.py`, all ImageNet and [SatlasPretrain](https://openaccess.thecvf.com/content/ICCV2023/html/Bastani_SatlasPretrain_A_Large-Scale_Dataset_for_Remote_Sensing_Image_Understanding_ICCV_2023_paper.html) pretrained weights see `NAS: Remote/RSFM/LXQ/swinv2`. Note: do not change the pretrained weight name.
- `2024/06/22`: Add [Cross-ScaleMAE](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3fadcbd0437f4717723ff3f6f7216800-Abstract-Conference.html) in `model/backbone/cross_scalemae`, including two versions based on timm (official implement) and our RSFM, by [Jing Zhang](https://github.com/Jerry-jing).
- `2024/06/22`: Add [SatMAE++](https://arxiv.org/abs/2403.05419) in `model/backbone/satmae_pp`, including two versions based on timm (official implement) and our RSFM, by [Jiamin Cao](https://github.com/JMcarrot).
- `2024/06/22`: Add [ScaleMAE](https://arxiv.org/abs/2212.14532) in `model/backbone/scalemae`, including two versions based on timm (official implement) and our RSFM, by [Chenyue Che](https://github.com/chenyueche).
- `2024/06/19`: Unify codes for semantic segmentation and change detection.
- `2024/06/14`: Add [ViT](https://arxiv.org/abs/2010.11929) family models in `model/backbone/vision_transformer.py`, all ImageNet pretrained weights see `NAS: Remote/RSFM/LXQ/vit`.
- `2024/06/14`: Add [VMamba](https://arxiv.org/abs/2401.10166) family models in `model/backbone/vmamba/vmamba.py`, all ImageNet pretrained weights see `NAS: Remote/RSFM/LXQ/vmamba`, by [Qin Ma](https://github.com/chunbai1).
- `2024/06/13`: Add [FocalNet](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1b08f585b0171b74d1401a5195e986f1-Abstract-Conference.html) family models in `model/backbone/focalnet.py`, all ImageNet pretrained weights see `NAS: Remote/RSFM/LXQ/focalnet`.
- `2024/06/07`: Add [FPN](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html), [PAFPN](https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Path_Aggregation_Network_CVPR_2018_paper.html), [BiFPN](https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html) in `model/module/fpn.py`.
- `2024/06/07`: Add [SemanticFPN](https://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Feature_Pyramid_Networks_CVPR_2019_paper.html) in `model/decoder/semantic_fpn.py`, config modification see `configs/DeepGlobe_SemanticFPN.yaml`.
- `2024/06/07`: Add [InternImage](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_InternImage_Exploring_Large-Scale_Vision_Foundation_Models_With_Deformable_Convolutions_CVPR_2023_paper.html) family models in `model/backbone/internimage.py`, all ImageNet pretrained weights see `NAS: Remote/RSFM/LXQ/internimage`.
- `2024/06/07`: Add [UniRepLKNet](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_UniRepLKNet_A_Universal_Perception_Large-Kernel_ConvNet_for_Audio_Video_Point_CVPR_2024_paper.html) family models in `model/backbone/unireplknet.py`, all ImageNet pretrained weights see `NAS: Remote/RSFM/LXQ/unireplknet`.
- `2024/06/06`: Add [ConvNeXt](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html) family models in `model/backbone/convnext.py`, all ImageNet pretrained weights see `NAS: Remote/RSFM/LXQ/convnext`.

## Logging
- `2024/06/24`: Support storage of normalized named weights, which needs to save pretrained weights like `./pretrained_weights/imagenet/model_name/*.pth` or `./pretrained_weights/satmae_pp/*.pth`.
- `2024/06/22`: The previous ViT models are initially built using the default `img_size=224`, which is not rigorous, yet no bugs are reported. The updated `builder.py` supports passing in the corresponding `img_size` for the ViT models based on the actual training size.
- `2024/06/12`: Support selective input channel dimensions for multi-spectral data, by [Xinyu Liu](https://github.com/xxxxyliu) and [Jiamin Cao](https://github.com/JMcarrot).
- `2024/06/11`: All backbone files support auto reset input channels and load aligned pretrained weights.
- `2024/06/07`: Since the specific value of `embed_dim` has been specified at each backbone file, there is no need to pass `embed_dim` in `model_info.py` to `builder.py`, which all involved has been removed; The missing `**kwargs` in each backbone file has been added; The changes don't affect the trained model for inference.

## Installation
```shell
# create environment rsfm 
conda create -n rsfm python=3.10 -y
conda activate rsfm
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
mim install mmcv-full==1.7.0
# for usage of InternImage
cd rsfm/backbone/internimage/ops_dcnv3
sh make.sh
python test.py
cd ../../../..
# for usage of VMamba
pip install triton fvcore
cd rsfm/backbone/vmamba/kernels/selective_scan && pip install .
cd ../../../../..
# for usage of RIS, VG, OD
pip install tokenizers h5py
pip install pytorch_pretrained_bert
```

## Usage
### Training
```shell
bash scripts/train_dist.sh <num gpus> <port>
# for example: bash scripts/train_dist.sh 4 10000
```

### Validation
```shell
# Normal validation
bash scripts/eval_dist.sh <num gpus> <port> --weight-path <path/to/your/trained/weight>

# Using test-time augmentation (TTA), including multi-scale (x1.0, x1.125, x1.25, x1.375, x1.5) augs with horizontal flipping
bash scripts/eval_dist.sh <num gpus> <port> --weight-path <path/to/your/trained/weight> --tta
```

### Prediction
```shell
# Normal prediction
bash scripts/eval_dist.sh <num gpus> <port> --task predict --weight-path <path/to/your/trained/weight> --save-path <path/to/dir/you/want/save>

# Using test-time augmentation (TTA)
bash scripts/eval_dist.sh <num gpus> <port> --task predict --weight-path <path/to/your/trained/weight> --save-path <path/to/dir/you/want/save> --tta
```
