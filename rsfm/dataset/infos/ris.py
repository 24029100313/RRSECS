ris_datasets_info = {

    'RRSIS-D':
        {'data_root': '/root/lxq/RSFM_RIS_Datasets/Optical/RRSIS-D',
         'classes_name':
             ['background', 'foreground'],
         'img_suffix': '.jpg',
         'mask_suffix': '.png',
         'num_bands': 3,  # number of channels of your input data
         'need_bands': False,  # use for multi-spectral data to specify order of input channels, like [2, 1, 0]
         'num_classes': 2,
         'reduce_zero_label': False,  # True represents your label starts from 1, else 0
         'vision_task': 'referring image segmentation',
         'training_size': 800,
         'color_map':
             [[255, 255, 255], [255, 0, 0]],
         'num_train': 13921,
         'num_val': 3481,
         'modality': 'optical',
         'source': 'CVPR 2024'},

    'RefSegRS':
        {'data_root': '/root/lxq/RSFM_RIS_Datasets/Optical/RefSegRS',
         'classes_name':
             ['background', 'foreground'],
         'img_suffix': '.tif',
         'mask_suffix': '.tif',
         'num_bands': 3,  # number of channels of your input data
         'need_bands': False,  # use for multi-spectral data to specify order of input channels, like [2, 1, 0]
         'num_classes': 2,
         'reduce_zero_label': False,  # True represents your label starts from 1, else 0
         'vision_task': 'referring image segmentation',
         'training_size': 512,
         'color_map':
             [[255, 255, 255], [255, 0, 0]],
         'num_train': 2603,
         'num_val': 1817,
         'modality': 'optical',
         'source': 'TGRS 2024'},

    'RefCOCO':
        {'data_root': '/root/lxq/RSFM_RIS_Datasets/Natural/RefCOCO',
         'classes_name':
             ['background', 'foreground'],
         'img_suffix': '.jpg',
         'mask_suffix': '.png',
         'num_bands': 3,  # number of channels of your input data
         'need_bands': False,  # use for multi-spectral data to specify order of input channels, like [2, 1, 0]
         'num_classes': 2,
         'reduce_zero_label': False,  # True represents your label starts from 1, else 0
         'vision_task': 'referring image segmentation',
         'training_size': 480,
         'color_map':
             [[255, 255, 255], [255, 0, 0]],
         'num_train': 42404,
         'num_val': 3811,
         'num_testA': 1975,
         'num_testB': 1810,
         'modality': 'natural',
         'source': 'Modeling Context in Referring Expressions, ECCV 2016'},

    'RefCOCO+':
        {'data_root': '/root/lxq/RSFM_RIS_Datasets/Natural/RefCOCO+',
         'classes_name':
             ['background', 'foreground'],
         'img_suffix': '.jpg',
         'mask_suffix': '.png',
         'num_bands': 3,  # number of channels of your input data
         'need_bands': False,  # use for multi-spectral data to specify order of input channels, like [2, 1, 0]
         'num_classes': 2,
         'reduce_zero_label': False,  # True represents your label starts from 1, else 0
         'vision_task': 'referring image segmentation',
         'training_size': 480,
         'color_map':
             [[255, 255, 255], [255, 0, 0]],
         'num_train': 42278,
         'num_val': 3805,
         'num_testA': 1975,
         'num_testB': 1798,
         'modality': 'natural',
         'source': 'Modeling Context in Referring Expressions, ECCV 2016'},

    'G-Ref-u':
        {'data_root': '/root/lxq/RSFM_RIS_Datasets/Natural/G-Ref-u',
         'classes_name':
             ['background', 'foreground'],
         'img_suffix': '.jpg',
         'mask_suffix': '.png',
         'num_bands': 3,  # number of channels of your input data
         'need_bands': False,  # use for multi-spectral data to specify order of input channels, like [2, 1, 0]
         'num_classes': 2,
         'reduce_zero_label': False,  # True represents your label starts from 1, else 0
         'vision_task': 'referring image segmentation',
         'training_size': 480,
         'color_map':
             [[255, 255, 255], [255, 0, 0]],
         'num_train': 42226,
         'num_val': 2573,
         'num_test': 5023,
         'modality': 'natural',
         'source': 'Generation and Comprehension of Unambiguous Object Descriptions, CVPR 2016'},

    'G-Ref-g':
        {'data_root': '/root/lxq/RSFM_RIS_Datasets/Natural/G-Ref-g',
         'classes_name':
             ['background', 'foreground'],
         'img_suffix': '.jpg',
         'mask_suffix': '.png',
         'num_bands': 3,  # number of channels of your input data
         'need_bands': False,  # use for multi-spectral data to specify order of input channels, like [2, 1, 0]
         'num_classes': 2,
         'reduce_zero_label': False,  # True represents your label starts from 1, else 0
         'vision_task': 'referring image segmentation',
         'training_size': 480,
         'color_map':
             [[255, 255, 255], [255, 0, 0]],
         'num_train': 44822,
         'num_val': 5000,
         'modality': 'natural',
         'source': 'Generation and Comprehension of Unambiguous Object Descriptions, CVPR 2016'},

}