cls_datasets_infos = {

    'EuroSAT':
        {'data_root': '/root/lxq/RSFM_CLS_Datasets/Optical/EuroSAT',
         'classes_name':
             ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop',
              'Residential', 'River', 'SeaLake'],
         'img_suffix': '.jpg',
         'num_bands': 3,
         'need_bands': False,
         'num_classes': 10,
         'vision_task': 'scene classification',
         'training_size': 64,
         'num_train': 10192,
         'num_val': 5568,
         'modality': 'optical',
         'source': 'A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection, RS 2020'},

    'Million-AID-mini':
        {'data_root': '/root/lxq/RSFM_CLS_Datasets/Million-AID-mini',
         'classes_name':
             ['apartment', 'apron', 'bare_land', 'baseball_field', 'basketball_court', 'beach', 'bridge', 'cemetery',
              'church', 'commercial_area', 'dam', 'desert', 'detached_house', 'dry_field', 'forest', 'golf_course',
              'greenhouse', 'ground_track_field', 'helipad', 'ice_land', 'intersection', 'island', 'lake', 'meadow',
              'mine', 'mobile_home_park', 'oil_field', 'orchard', 'paddy_field', 'parking_lot', 'pier', 'quarry',
              'railway', 'river', 'road', 'rock_land', 'roundabout', 'runway', 'solar_power_plant',
              'sparse_shrub_land', 'stadium', 'storage_tank', 'substation', 'swimming_pool', 'tennis_court',
              'terraced_field', 'train_station', 'viaduct', 'wastewater_plant', 'wind_turbine', 'works'],
         'img_suffix': '.jpg',
         'num_bands': 3,
         'need_bands': False,
         'num_classes': 51,
         'vision_task': 'scene classification',
         'training_size': 224,
         'num_train': 9000,
         'num_val': 1000,
         'modality': 'optical',
         'source': 'On Creating Benchmark Dataset for Aerial Image Interpretation: Reviews, Guidances, and Million-AID, JSTARS 2021'},

}