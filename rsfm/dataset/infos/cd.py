cd_datasets_info = {

    'LEVIR-CD+':
        {'data_root': '/root/lxq/RSFM_CD_Datasets/Optical/LEVIR-CD+',
         'classes_name':
             ['unchanged', 'changed'],
         'img_suffix': '.png',
         'mask_suffix': '.png',
         'num_bands': 3,
         'need_bands': False,
         'num_classes': 2,
         'reduce_zero_label': False,
         'vision_task': 'change detection',
         'training_size': 256,
         'color_map':
             [[0, 0, 0], [255, 255, 255]],
         'num_train': 10192,
         'num_val': 5568,
         'modality': 'optical',
         'source': 'A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection, RS 2020'},

    'Hi-CNA':
        {'data_root': '/root/lxq/RSFM_CD_Datasets/Infrared/Hi-CNA',
         'classes_name':
             ['unchanged', 'changed'],
         'color_map':
             [[0, 0, 0], [255, 255, 255]],
         'img_suffix': '.tif',
         'mask_suffix': '.png',
         'num_train': 4080,
         'num_val': 1358,
         'num_bands': 1,
         'need_bands': False,
         'num_classes': 2,
         'reduce_zero_label': False,
         'modality': 'infrared',
         'vision_task': 'change detection',
         'training_size': 512,
         'source': 'Identifying cropland non-agriculturalization with high representational consistency from '
                   'bi-temporal high-resolution remote sensing images: From benchmark datasets to real-world application, ISPRS 2024'},

    'S1GFloods':
        {'data_root': '/root/lxq/RSFM_CD_Datasets/SAR/S1GFloods',
         'classes_name':
             ['unchanged', 'changed'],
         'img_suffix': '.png',
         'mask_suffix': '.png',
         'num_bands': 1,
         'need_bands': False,
         'num_classes': 2,
         'reduce_zero_label': False,
         'vision_task': 'change detection',
         'training_size': 256,
         'color_map':
             [[0, 0, 0], [255, 255, 255]],
         'num_train': 4300,
         'num_val': 1060,
         'modality': 'sar',
         'source': 'DAM-Net: Flood detection from SAR imagery using differential attention metric-based vision transformers, ISPRS 2024'},

    'OSCD':
        {'data_root': '/root/lxq/RSFM_CD_Datasets/Multispectral/OSCD',
         'classes_name':
             ['unchanged', 'changed'],
         'img_suffix': '.png',
         'mask_suffix': '.png',
         'num_bands': 3,
         'need_bands': False,
         'num_classes': 2,
         'reduce_zero_label': False,
         'vision_task': 'change detection',
         'training_size': 128,
         'color_map':
             [[0, 0, 0], [255, 255, 255]],
         'num_train': 496,
         'num_val': 230,
         'modality': 'multi-spectral',
         'source': 'Urban change detection for multispectral earth observation using convolutional neural networks, IGARSS 2018'},

}