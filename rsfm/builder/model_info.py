# Hyper-parameters setting for various types of backbone


encoder_info = {

    'resnet50': {'enc_out_dims': [256, 512, 1024, 2048]},
    'resnet101': {'enc_out_dims': [256, 512, 1024, 2048]},

    'mit_b0': {'enc_out_dims': [32, 64, 160, 256]},
    'mit_b1': {'enc_out_dims': [64, 128, 320, 512]},
    'mit_b2': {'enc_out_dims': [64, 128, 320, 512]},
    'mit_b3': {'enc_out_dims': [64, 128, 320, 512]},
    'mit_b4': {'enc_out_dims': [64, 128, 320, 512]},
    'mit_b5': {'enc_out_dims': [64, 128, 320, 512]},

    'pvtv2_b0': {'enc_out_dims': [32, 64, 160, 256]},
    'pvtv2_b1': {'enc_out_dims': [64, 128, 320, 512]},
    'pvtv2_b2': {'enc_out_dims': [64, 128, 320, 512]},
    'pvtv2_b3': {'enc_out_dims': [64, 128, 320, 512]},
    'pvtv2_b4': {'enc_out_dims': [64, 128, 320, 512]},
    'pvtv2_b5': {'enc_out_dims': [64, 128, 320, 512]},

    'van_b0': {'enc_out_dims': [32, 64, 160, 256]},
    'van_b1': {'enc_out_dims': [64, 128, 320, 512]},
    'van_b2': {'enc_out_dims': [64, 128, 320, 512]},
    'van_b3': {'enc_out_dims': [64, 128, 320, 512]},
    'van_b4': {'enc_out_dims': [64, 128, 320, 512]},
    'van_b5': {'enc_out_dims': [96, 192, 480, 768]},
    'van_b6': {'enc_out_dims': [96, 192, 384, 768]},

    'lsknet_tiny': {'enc_out_dims': [32, 64, 160, 256]},
    'lsknet_small': {'enc_out_dims': [64, 128, 320, 512]},

    'casvit_xs': {'enc_out_dims': [48, 56, 112, 220]},
    'casvit_s': {'enc_out_dims': [48, 64, 128, 256]},
    'casvit_m': {'enc_out_dims': [64, 96, 192, 384]},
    'casvit_t': {'enc_out_dims': [96, 128, 256, 512]},

    'swin_tiny': {'enc_out_dims': [96, 192, 384, 768]},
    'swin_small': {'enc_out_dims': [96, 192, 384, 768]},
    'swin_base': {'enc_out_dims': [128, 256, 512, 1024]},
    'swin_base_gfm_w6': {'enc_out_dims': [128, 256, 512, 1024]},
    'swin_large': {'enc_out_dims': [192, 384, 768, 1536]},

    'swinv2_tiny': {'enc_out_dims': [96, 192, 384, 768]},
    'swinv2_small': {'enc_out_dims': [96, 192, 384, 768]},
    'swinv2_base': {'enc_out_dims': [128, 256, 512, 1024]},
    'swinv2_large': {'enc_out_dims': [192, 384, 768, 1536]},

    'cswin_tiny': {'enc_out_dims': [64, 128, 256, 512]},
    'cswin_small': {'enc_out_dims': [64, 128, 256, 512]},
    'cswin_base': {'enc_out_dims': [96, 192, 384, 768]},
    'cswin_large': {'enc_out_dims': [144, 288, 576, 1152]},

    'convnext_tiny': {'enc_out_dims': [96, 192, 384, 768]},
    'convnext_small': {'enc_out_dims': [96, 192, 384, 768]},
    'convnext_base': {'enc_out_dims': [128, 256, 512, 1024]},
    'convnext_large': {'enc_out_dims': [192, 384, 768, 1536]},
    'convnext_xlarge': {'enc_out_dims': [256, 512, 1024, 2048]},

    'unireplknet_tiny': {'enc_out_dims': [80, 160, 320, 640]},
    'unireplknet_small': {'enc_out_dims': [96, 192, 384, 768]},
    'unireplknet_base': {'enc_out_dims': [128, 256, 512, 1024]},
    'unireplknet_large': {'enc_out_dims': [192, 384, 768, 1536]},
    'unireplknet_xlarge': {'enc_out_dims': [256, 512, 1024, 2048]},

    'internimage_tiny': {'enc_out_dims': [64, 128, 256, 512]},
    'internimage_small': {'enc_out_dims': [80, 160, 320, 640]},
    'internimage_base': {'enc_out_dims': [112, 224, 448, 896]},
    'internimage_large': {'enc_out_dims': [160, 320, 640, 1280]},
    'internimage_xlarge': {'enc_out_dims': [192, 384, 768, 1536]},
    'internimage_huge': {'enc_out_dims': [320, 640, 1280, 2560]},

    'focalnet_tiny': {'enc_out_dims': [96, 192, 384, 768]},
    'focalnet_small': {'enc_out_dims': [96, 192, 384, 768]},
    'focalnet_base': {'enc_out_dims': [128, 256, 512, 1024]},
    'focalnet_large': {'enc_out_dims': [192, 384, 768, 1536]},
    'focalnet_xlarge': {'enc_out_dims': [256, 512, 1024, 2048]},

    'vmamba_tiny': {'enc_out_dims': [96, 192, 384, 768]},
    'vmamba_small': {'enc_out_dims': [96, 192, 384, 768]},
    'vmamba_base': {'enc_out_dims': [128, 256, 512, 1024]},

    'vit_base': {'enc_out_dims': [768, 768, 768, 768]},
    'vit_base_patch16_spectralgpt': {'enc_out_dims': [768, 768, 768, 768]},
    'vit_base_patch16_spatsigma': {'enc_out_dims': [768, 768, 768, 768]},
    'vit_base_patch16_hypersigma': {'enc_out_dims': [768, 768, 768, 768]},

    'vit_large': {'enc_out_dims': [1024, 1024, 1024, 1024]},
    'vit_large_patch16_satmae_pp_timm': {'enc_out_dims': [1024, 1024, 1024, 1024]},
    'vit_large_patch16_satmae_pp_rsfm': {'enc_out_dims': [1024, 1024, 1024, 1024]},
    'vit_large_patch16_scalemae_timm': {'enc_out_dims': [1024, 1024, 1024, 1024]},
    'vit_large_patch16_scalemae_rsfm': {'enc_out_dims': [1024, 1024, 1024, 1024]},
    'vit_large_patch16_cross_scalemae_timm': {'enc_out_dims': [1024, 1024, 1024, 1024]},
    'vit_large_patch16_cross_scalemae_rsfm': {'enc_out_dims': [1024, 1024, 1024, 1024]},
    'vit_large_patch16_spatsigma': {'enc_out_dims': [1024, 1024, 1024, 1024]},
    'vit_large_patch16_hypersigma': {'enc_out_dims': [1024, 1024, 1024, 1024]},

    'dinov2_small': {'enc_out_dims': [384, 384, 384, 384]},
    'dinov2_base': {'enc_out_dims': [768, 768, 768, 768]},
    'dinov2_large': {'enc_out_dims': [1024, 1024, 1024, 1024]},
    'dinov2_giant': {'enc_out_dims': [1536, 1536, 1536, 1536]},

    'beit_base': {'enc_out_dims': [768, 768, 768, 768]},
    'beit_large': {'enc_out_dims': [1024, 1024, 1024, 1024]},

    'beitv2_base': {'enc_out_dims': [768, 768, 768, 768]},
    'beitv2_large': {'enc_out_dims': [1024, 1024, 1024, 1024]},

    'iformer_small': {'enc_out_dims': [96, 192, 320, 384]},
    'iformer_base': {'enc_out_dims': [96, 192, 384, 512]},
    'iformer_large': {'enc_out_dims': [96, 192, 448, 640]},

    'inceptionnext_tiny': {'enc_out_dims': [96, 192, 384, 768]},
    'inceptionnext_small': {'enc_out_dims': [96, 192, 384, 768]},
    'inceptionnext_base': {'enc_out_dims': [128, 256, 512, 1024]},

    'transnext_micro': {'enc_out_dims': [48, 96, 192, 384]},
    'transnext_tiny': {'enc_out_dims': [72, 144, 288, 576]},
    'transnext_small': {'enc_out_dims': [72, 144, 288, 576]},
    'transnext_base': {'enc_out_dims': [96, 192, 384, 768]},

    'mvitv2_tiny': {'enc_out_dims': [96, 192, 384, 768]},
    'mvitv2_small': {'enc_out_dims': [96, 192, 384, 768]},
    'mvitv2_base': {'enc_out_dims': [96, 192, 384, 768]},
    'mvitv2_large': {'enc_out_dims': [144, 288, 576, 1152]},
    'mvitv2_huge': {'enc_out_dims': [192, 384, 768, 1536]},

    'eva2_base': {'enc_out_dims': [768, 768, 768, 768]},
    'eva2_large': {'enc_out_dims': [1024, 1024, 1024, 1024]},

    'wavevit_small': {'enc_out_dims': [64, 128, 320, 448]},
    'wavevit_base': {'enc_out_dims': [64, 128, 320, 512]},
    'wavevit_large': {'enc_out_dims': [96, 192, 384, 512]},

    'gfnet_tiny': {'enc_out_dims': [64, 128, 256, 512]},
    'gfnet_small': {'enc_out_dims': [96, 192, 384, 768]},
    'gfnet_base': {'enc_out_dims': [96, 192, 384, 768]},

    'starnet_s1': {'enc_out_dims': [24, 48, 96, 192]},
    'starnet_s2': {'enc_out_dims': [32, 64, 128, 256]},
    'starnet_s3': {'enc_out_dims': [32, 64, 128, 256]},
    'starnet_s4': {'enc_out_dims': [32, 64, 128, 256]},

    'moganet_xtiny': {'enc_out_dims': [32, 64, 96, 192]},
    'moganet_tiny': {'enc_out_dims': [32, 64, 128, 256]},
    'moganet_small': {'enc_out_dims': [64, 128, 320, 512]},
    'moganet_base': {'enc_out_dims': [64, 160, 320, 512]},
    'moganet_large': {'enc_out_dims': [64, 160, 320, 640]},
    'moganet_xlarge': {'enc_out_dims': [96, 192, 480, 960]},

    'mambaout_femto': {'enc_out_dims': [48, 96, 192, 288]},
    'mambaout_kobe': {'enc_out_dims': [48, 96, 192, 288]},
    'mambaout_tiny': {'enc_out_dims': [96, 192, 384, 576]},
    'mambaout_small': {'enc_out_dims': [96, 192, 384, 576]},
    'mambaout_base': {'enc_out_dims': [128, 256, 512, 768]},

    'mlla_tiny': {'enc_out_dims': [64, 128, 256, 512]},
    'mlla_small': {'enc_out_dims': [64, 128, 256, 512]},
    'mlla_base': {'enc_out_dims': [96, 192, 384, 768]},

    'convformer_s18': {'enc_out_dims': [64, 128, 320, 512]},
    'convformer_s36': {'enc_out_dims': [64, 128, 320, 512]},
    'convformer_m36': {'enc_out_dims': [96, 192, 384, 576]},
    'convformer_b36': {'enc_out_dims': [128, 256, 512, 768]},

    'caformer_s18': {'enc_out_dims': [64, 128, 320, 512]},
    'caformer_s36': {'enc_out_dims': [64, 128, 320, 512]},
    'caformer_m36': {'enc_out_dims': [96, 192, 384, 576]},
    'caformer_b36': {'enc_out_dims': [128, 256, 512, 768]},

    'poolformerv2_s12': {'enc_out_dims': [64, 128, 320, 512]},
    'poolformerv2_s24': {'enc_out_dims': [64, 128, 320, 512]},
    'poolformerv2_s36': {'enc_out_dims': [64, 128, 320, 512]},
    'poolformerv2_m36': {'enc_out_dims': [96, 192, 384, 768]},
    'poolformerv2_m48': {'enc_out_dims': [96, 192, 384, 768]},

}


decoder_info = {

    # seg / cd decoder
    'SegFormerHead': {'dec_out_dim': 256},
    'UMixFormerHead': {'dec_out_dim': 256},
    'LightHamHead': {'dec_out_dim': 256},
    'UNetHead': {'dec_out_dim': 64},
    'UNetPPHead': {'dec_out_dim': 64},
    'AttnUNetHead':  {'dec_out_dim': 64},
    'UNetFormerHead': {'dec_out_dim': 256},
    'UNetv2Head': {'dec_out_dim': 256},
    'UPerHead': {'dec_out_dim': 512},
    'FPNHead': {'dec_out_dim': 512},
    'DeepLabV3PlusHead': {'dec_out_dim': 256},
    'Mask2FormerHead': {'dec_out_dim': 256},
    'DPTHead': {'dec_out_dim': 256},
    'EMCADHead': {'dec_out_dim': 256},
    'ABCNetHead': {'dec_out_dim': 256},
    'AerialFormerHead': {'dec_out_dim': 96},
    'BANetHead': {'dec_out_dim': 256},
    'DCSwinHead': {'dec_out_dim': 96},
    'DuATHead': {'dec_out_dim': 64},
    'CASCADEHead': {'dec_out_dim': 64},
    'PolypHead': {'dec_out_dim': 32},

    # cd-specific decoder
    'BITCDHead': {'dec_out_dim': 32},
    'MambaBCDHead': {'dec_out_dim': 128},

    # ris decoder
    'LAVTHead': {'dec_out_dim': 512},
    'LGCEHead': {'dec_out_dim': 512},
    'RMSINHead': {'dec_out_dim': 512},
    'CGFormerHead': {'dec_out_dim': 512},
    'ReMamberHead': {'dec_out_dim': 512},
    'MCTHead': {'dec_out_dim': 512},
    'MagNetHead': {'dec_out_dim': 256},
    'ReLAHead': {'dec_out_dim': 256},
    'DMMIHead': {'dec_out_dim': 256},
    'RefSegformerHead': {'dec_out_dim': 256},
    'CrossVLTHead': {'dec_out_dim': 512},
    'CCFormerRISHead': {'dec_out_dim': 256},

    # vg decoder
    'TransVGHead': {'dec_out_dim': 256},
    'VLTVGHead': {'dec_out_dim': 256},
    'LQVGHead': {'dec_out_dim': 256},
    'TransCPHead': {'dec_out_dim': 256},
    'PseudoQHead': {'dec_out_dim': 256},
    'QRNetHead': {'dec_out_dim': 256},
    'DynamicMDETRHead': {'dec_out_dim': 256},
    'LPVAHead': {'dec_out_dim': 256},

    # recs decoder
    'CCFormerHead': {'dec_out_dim': 256},
    'MRLNHead': {'dec_out_dim': 512},
    'MCNHead': {'dec_out_dim': 512},
    'RefTRHead': {'dec_out_dim': 256},

    # det decoder:
    'DETRHead': {'dec_out_dim': 256},
    'DeformableDETRHead': {'dec_out_dim': 256},
    'DINOHead': {'dec_out_dim': 256},

}


neck_info = {

    'fpn': {'out_channels': 256},
    'pafpn': {'out_channels': 256},
    'bifpn': {'out_channels': 256},
    'QMF': {'out_channels': 256},
    'MSCAB': {'out_channels': 256},
    'vg_neck': {'out_channels': 256},

    'multilevel_neck': {'out_channels': 768}, # Change has no effect, set automatically by the encoder.
    'p2h': {'out_channels': 768}, # Change has no effect, set automatically by the encoder.

}


language_info = {

    'bert-base-uncased': {'hidden_dim': 768},
    'bert-large-uncased': {'hidden_dim': 1024},
    'roberta-base': {'hidden_dim': 768},
    'autotinybert-s1': {'hidden_dim': 564},
    'autotinybert-s2': {'hidden_dim': 396},
    'autotinybert-s3': {'hidden_dim': 432},
    'autotinybert-s4': {'hidden_dim': 320},
    'autotinybert-kd-s1': {'hidden_dim': 564},
    'autotinybert-kd-s2': {'hidden_dim': 324},
    'autotinybert-kd-s3': {'hidden_dim': 280},
    'autotinybert-kd-s4': {'hidden_dim': 256},

}
