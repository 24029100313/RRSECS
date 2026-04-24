seg_datasets_info = {

    'Potsdam':
        {'data_root': '/root/lxq/RSFM_Seg_Datasets/Optical/Potsdam',
         'classes_name':
             ['impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter'],
         'img_suffix': '.png',
         'mask_suffix': '.png',
         'num_bands': 3,  # number of channels of your input data
         'need_bands': False,  # use for multi-spectral data to specify order of input channels, like [2, 1, 0]
         'num_classes': 6,
         'reduce_zero_label': True,  # True represents your label starts from 1, else 0
         'vision_task': 'semantic segmentation',
         'training_size': 512,
         'color_map':
             [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255]],
         'num_train': 3456,
         'num_val': 2016,
         'modality': 'optical',
         'source': 'ISPRS'},

    'Potsdam_woclutter':
        {'data_root': '/root/lxq/RSFM_Seg_Datasets/Optical/Potsdam_woclutter',
         'classes_name':
             ['impervious_surface', 'building', 'low_vegetation', 'tree', 'car'],
         'img_suffix': '.png',
         'mask_suffix': '.png',
         'num_bands': 3,  # number of channels of your input data
         'need_bands': False,  # use for multi-spectral data to specify order of input channels, like [2, 1, 0]
         'num_classes': 5,
         'reduce_zero_label': True,  # True represents your label starts from 1, else 0
         'vision_task': 'semantic segmentation',
         'training_size': 512,
         'color_map':
             [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255]],
         'num_train': 3456,
         'num_val': 2016,
         'modality': 'optical',
         'source': 'ISPRS'},

    'DeepGlobe':
        {'data_root': '/root/lxq/RSFM_Seg_Datasets/Optical/DeepGlobe',
         'classes_name':
             ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown'],
         'img_suffix': '.jpg',
         'mask_suffix': '.png',
         'num_bands': 3,  # number of channels of your input data
         'need_bands': False, # use for multi-spectral data to specify order of input channels, like [2, 1, 0]
         'num_classes': 7,
         'reduce_zero_label': False,  # True represents your label starts from 1, else 0
         'vision_task': 'semantic segmentation',
         'training_size': 512,
         'color_map':
             [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]],
         'num_train': 5760,
         'num_val': 1467,
         'modality': 'optical',
         'source': 'DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images, CVPRW 2018'},

    'LoveDA':
         {'data_root': '/root/lxq/RSFM_Seg_Datasets/Optical/LoveDA',
          'classes_name':
              ['background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural'],
          'img_suffix': '.png',
          'mask_suffix': '.png',
          'num_bands':3,
          'need_bands': False,
          'num_classes': 7,
          'reduce_zero_label': False,
          'vision_task': 'semantic segmentation',
          'training_size': 512,
          'color_map':
              [[0, 0, 0], [255, 0, 0], [255, 255, 0], [0, 0, 255], [159, 129, 183], [0, 255, 0], [255, 195, 128]],
          'num_train': 2522,
          'num_val': 1669,
          'modality': 'optical',
          'source': 'LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation, NIPS 2021'},

    'WHU_OPT_SAR':
         {'data_root': '/root/lxq/RSFM_Seg_Datasets/SAR/WHU_OPT_SAR',
          'classes_name':
              ['farmland', 'city', 'village', 'water', 'forest', 'road', 'others'],
          'img_suffix': '.tif',
          'mask_suffix': '.tif',
          'num_bands':1,
          'need_bands': False,
          'num_classes': 7,
          'reduce_zero_label': True,
          'vision_task': 'semantic segmentation',
          'training_size': 512,
          'color_map':
              [[201, 101, 2], [248, 2, 0], [255, 242, 3], [0, 6, 191], [84, 164, 4], [112, 232, 253], [149, 102, 151]],
          'num_train': 7040,
          'num_val': 1760,
          'modality': 'sar',
          'source': 'MCANet: A joint semantic segmentation framework of optical and SAR images for land use classification, JAG 2022'},

    'DFC24_T1':
         {'data_root': '/root/lxq/RSFM_Seg_Datasets/SAR/DFC24_T1',
          'classes_name':
              ['non-water', 'water'],
          'img_suffix': '.tif',
          'mask_suffix': '.png',
          'num_bands':1,
          'need_bands': False,
          'num_classes': 2,
          'reduce_zero_label': False,
          'vision_task': 'semantic segmentation',
          'training_size': 512,
          'color_map':
              [[166, 166, 166], [9, 48, 107]],
          'num_train': 1304,
          'num_val': 327,
          'modality': 'sar',
          'source': '2024 IEEE GRSS Data Fusion Contest. Online: https://www.grss-ieee.org/technical-committees/image-analysis-and-data-fusion/'},

    'Agriculture_Vision':
        {'data_root': '/root/lxq/RSFM_Seg_Datasets/Infrared/agriculture_vision',
         'classes_name':
             ['background', 'double_plant', 'drydown', 'endrow', 'nutrient_deficiency', 'planter_skip', 'water', 'waterway', 'weed_cluster'],
         'img_suffix': '.png',
         'mask_suffix': '.png',
         'num_bands':1,
         'need_bands': False,
         'num_classes': 9,
         'reduce_zero_label': False,
         'vision_task': 'semantic segmentation',
         'training_size': 256,
         'color_map':
             [[64, 64, 64], [23, 190, 207], [32, 119, 180], [148, 103, 189], [43, 160, 44], [127, 127, 127],
              [214, 39, 40], [140, 86, 75], [255, 127, 14]],
         'num_train': 56944,
         'num_val': 18334,
         'modality': 'infrared',
         'source': 'Agriculture-Vision: A Large Aerial Image Database for Agricultural Pattern Analysis, CVPR 2020'},

    'SPARCS':
        {'data_root': '/root/lxq/RSFM_Seg_Datasets/Multispectral/SPARCS',
         'classes_name':
             ['shadow', 'water', 'snow', 'land', 'cloud'],
         'img_suffix': '.tif',
         'mask_suffix': '.png',
         'num_bands':10,
         'need_bands': [2, 1, 0],
         'num_classes': 5,
         'reduce_zero_label': False,
         'vision_task': 'semantic segmentation',
         'training_size': 512,
         'color_map':
             [[0, 0, 0], [0, 0, 255], [0, 255, 255], [128, 128, 128], [255, 255, 255]],
         'num_train': 1024,
         'num_val': 256,
         'modality': 'multi-spectral',
         'source': 'Automated detection of cloud and cloud shadow in single-date Landsat imagery using neural '
                   'networks and spatial post-processing, RS 2014'},

    'SegMunich':
        {'data_root': '/root/lxq/RSFM_Seg_Datasets/Multispectral/SegMunich',
         'classes_name':
             ['background', 'arable_land', 'permanent_crops', 'pastures', 'forests', 'surface_water', 'shrub',
              'open_spaces', 'wetlands', 'mine', 'artificial_vegetation', 'urban_fabric', 'buildings'],
         'img_suffix': '.tif',
         'mask_suffix': '.tif',
         'num_bands': 10,
         'need_bands': [2, 1, 0],
         'num_classes': 13,
         'reduce_zero_label': False,
         'vision_task': 'semantic segmentation',
         'training_size': 128,
         'color_map':
             [[0, 0, 0], [249, 215, 128], [198, 129, 62], [185, 178, 106], [107, 184, 46],
              [112, 198, 211], [148, 198, 56], [133, 69, 34], [107, 106, 174],
              [112, 58, 144], [200, 137, 184], [230, 31, 24], [204, 156, 196]],
         'num_train': 7872,
         'num_val': 1974,
         'modality': 'multi-spectral',
         'source': 'SpectralGPT: Spectral foundation model, TPAMI 2024'},

}