decoder_cfg=dict(
    type='DecoderWithExtraEncoder',
    num_queries=1,
    query_dim=256,
    norm_dim = 256,
    return_intermediate=True,
    num_layers=6,
    layer=dict(
        type='MultiStageDecoderLayer',
        d_model=256,
        dim_feedforward=2048,
        dropout=0.,
        word_attn_args=dict(
            type='MultiheadAttention',
            embed_dim=256, num_heads=8, dropout=0.1,
        ),
        img_attn_args=dict(
            type='MultiheadAttention',
            embed_dim=256, num_heads=8, dropout=0.1,
        ),
        img_feat_chunk_num = 2,
    ),
    num_extra_layers=1,
    extra_layer=dict(
        type='DiscriminativeFeatEncLayer',
        d_model=256,
        img_query_with_pos=False,
        img2text_attn_args=dict(
            type='MultiheadAttention',
            embed_dim=256, num_heads=8, dropout=0.1
        ),
        img2textcond_attn_args=dict(
            type='MultiheadAttention',
            embed_dim=256, num_heads=8, dropout=0.1
        ),
        img2img_attn_args=dict(
            type='MHAttentionRPE',
            d_model=256, h=8, dropout=0.1
        ),
        vl_verify=dict(
            text_proj=dict(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1),
            img_proj=dict(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1),
            scale=1.0,
            sigma=0.5,
            pow=2.0,
        ),
    )
)