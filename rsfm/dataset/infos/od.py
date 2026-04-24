od_datasets_info = {

    'NWPU_VHR-10':
        {'data_root': '/root/lxq/RSFM_OD_Datasets/NWPU_VHR-10',
         'classes_name': ['airplane', 'ship', 'storage', 'baseball diamond', 'tennis court',
                          'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle'],
         'img_suffix': '.jpg',
         'num_bands': 3,  # number of channels of your input data
         'need_bands': False,  # use for multi-spectral data to specify order of input channels, like [2, 1, 0]
         'num_classes': 10,
         'reduce_zero_label': False,  # True represents your label starts from 1, else 0
         'vision_task': 'object detection',
         'training_size': 640,
         'color_map': [[255, 0, 0]],
         'num_train': 640,
         'num_val': 160,
         'modality': 'optical',
         'source': 'TGRS 2024'},

}