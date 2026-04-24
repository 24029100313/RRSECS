from .vmamba import VSSM


__all__ = ['vmamba_tiny', 'vmamba_small', 'vmamba_base']


class vmamba_tiny(VSSM):
    def __init__(self, **kwargs):
        super(vmamba_tiny, self).__init__(
            dims=96, depths=(2, 2, 5, 2), ssm_d_state=1, ssm_dt_rank='auto',
            ssm_ratio=2.0, ssm_conv=3, ssm_conv_bias=False, mlp_ratio=4.0,
            forward_type='v3_noz', downsample_version='v3', patchembed_version='v2',
            out_indices=(0, 1, 2, 3), drop_path_rate=0.2, patch_norm=True,
            ssm_init='v0', norm_layer='ln', patch_size=4, ssm_drop_rate=0.0, **kwargs)


class vmamba_small(VSSM):
    def __init__(self, **kwargs):
        super(vmamba_small, self).__init__(
            dims=96, depths=(2, 2, 15, 2), ssm_d_state=1, ssm_dt_rank='auto',
            ssm_ratio=2.0, ssm_conv=3, ssm_conv_bias=False, mlp_ratio=4.0,
            forward_type='v3_noz', downsample_version='v3', patchembed_version='v2',
            out_indices=(0, 1, 2, 3), drop_path_rate=0.3, patch_norm=True,
            ssm_init='v0', norm_layer='ln', patch_size=4, ssm_drop_rate=0.0, **kwargs)


class vmamba_base(VSSM):
    def __init__(self, **kwargs):
        super(vmamba_base, self).__init__(
            dims=128, depths=(2, 2, 15, 2), ssm_d_state=1, ssm_dt_rank='auto',
            ssm_ratio=2.0, ssm_conv=3, ssm_conv_bias=False, mlp_ratio=4.0,
            forward_type='v3_noz', downsample_version='v3', patchembed_version='v2',
            out_indices=(0, 1, 2, 3), drop_path_rate=0.6, patch_norm=True,
            ssm_init='v0', norm_layer='ln', patch_size=4, ssm_drop_rate=0.0, **kwargs)
