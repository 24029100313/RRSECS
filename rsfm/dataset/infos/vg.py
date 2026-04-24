vg_datasets_info = {

    'DIOR-RSVG':
        {'data_root': '/root/lxq/RSFM_VG_Datasets/Optical/DIOR-RSVG',
         'classes_name': ['object'],
         'img_suffix': '.jpg',
         'num_bands': 3,  # number of channels of your input data
         'need_bands': False,  # use for multi-spectral data to specify order of input channels, like [2, 1, 0]
         'num_classes': 1,
         'reduce_zero_label': False,  # True represents your label starts from 1, else 0
         'vision_task': 'visual grounding',
         'training_size': 640,
         'color_map': [[255, 0, 0]],
         'num_train_img_text_pairs': 30820,
         'num_val_img_text_pairs': 7500,
         'modality': 'optical',
         'source': 'TGRS 2024'},

    'OPT-RSVG':
        {'data_root': '/root/lxq/RSFM_VG_Datasets/Optical/OPT-RSVG',
         'classes_name': ['object'],
         'img_suffix': '.jpg',
         'num_bands': 3,  # number of channels of your input data
         'need_bands': False,  # use for multi-spectral data to specify order of input channels, like [2, 1, 0]
         'num_classes': 1,
         'reduce_zero_label': False,  # True represents your label starts from 1, else 0
         'vision_task': 'visual grounding',
         'training_size': 640,
         'color_map': [[255, 0, 0]],
         'num_train': 39162,
         'num_val': 9790,
         'modality': 'optical',
         'source': 'TGRS 2024'},

    'RefCOCO_VG':
        {'data_root': '/root/lxq/RSFM_RIS_Datasets/Natural/RefCOCO',
         'classes_name': ['object'],
         'img_suffix': '.jpg',
         'num_bands': 3,  # number of channels of your input data
         'need_bands': False,  # use for multi-spectral data to specify order of input channels, like [2, 1, 0]
         'num_classes': 2,
         'reduce_zero_label': False,  # True represents your label starts from 1, else 0
         'vision_task': 'visual grounding',
         'training_size': 640,
         'color_map': [[255, 0, 0]],
         'num_train': 42404,
         'num_val': 3811,
         'num_testA': 1975,
         'num_testB': 1810,
         'modality': 'natural',
         'source': 'Modeling Context in Referring Expressions, ECCV 2016'},

    'RefCOCO+_VG':
        {'data_root': '/root/lxq/RSFM_RIS_Datasets/Natural/RefCOCO+',
         'classes_name': ['object'],
         'img_suffix': '.jpg',
         'num_bands': 3,  # number of channels of your input data
         'need_bands': False,  # use for multi-spectral data to specify order of input channels, like [2, 1, 0]
         'num_classes': 2,
         'reduce_zero_label': False,  # True represents your label starts from 1, else 0
         'vision_task': 'visual grounding',
         'training_size': 640,
         'color_map': [[255, 0, 0]],
         'num_train': 42278,
         'num_val': 3805,
         'num_testA': 1975,
         'num_testB': 1798,
         'modality': 'natural',
         'source': 'Modeling Context in Referring Expressions, ECCV 2016'},

    'G-Ref-u_VG':
        {'data_root': '/root/lxq/RSFM_RIS_Datasets/Natural/G-Ref-u',
         'classes_name': ['object'],
         'img_suffix': '.jpg',
         'num_bands': 3,  # number of channels of your input data
         'need_bands': False,  # use for multi-spectral data to specify order of input channels, like [2, 1, 0]
         'num_classes': 2,
         'reduce_zero_label': False,  # True represents your label starts from 1, else 0
         'vision_task': 'visual grounding',
         'training_size': 640,
         'color_map': [[255, 0, 0]],
         'num_train': 42226,
         'num_val': 2573,
         'num_test': 5023,
         'modality': 'natural',
         'source': 'Generation and Comprehension of Unambiguous Object Descriptions, CVPR 2016'},

    'G-Ref-g_VG':
        {'data_root': '/root/lxq/RSFM_RIS_Datasets/Natural/G-Ref-g',
         'classes_name': ['object'],
         'img_suffix': '.jpg',
         'num_bands': 3,  # number of channels of your input data
         'need_bands': False,  # use for multi-spectral data to specify order of input channels, like [2, 1, 0]
         'num_classes': 2,
         'reduce_zero_label': False,  # True represents your label starts from 1, else 0
         'vision_task': 'visual grounding',
         'training_size': 640,
         'color_map': [[255, 0, 0]],
         'num_train': 44822,
         'num_val': 5000,
         'modality': 'natural',
         'source': 'Generation and Comprehension of Unambiguous Object Descriptions, CVPR 2016'},

}