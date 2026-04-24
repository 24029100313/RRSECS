import torch, einops
from torch import nn
from rsfm.module.ris_fusion.remamber_fusion.utils import VSSLayer, Linear2d, ImageTextCorr
from .utils import UpSample2D, Fusion, conv_layer



class ReMamberHead(nn.Module):
    '''
    from 'ReMamber: Referring Image Segmentation with Mamba Twister, ECCV 2024'
    '''
    def __init__(self,
                 in_channels=[128, 256, 512, 1024],
                 embedding_dim=512,
                 num_classes=2,
                 **kwargs):
        super(ReMamberHead, self).__init__()

        depths = [2, 4, 2, 2]
        dims = in_channels[::-1]
        num_layers = len(dims)

        self.text_guidance = nn.ModuleList()
        self.local_text_fusion = nn.ModuleList()
        self.multimodal_blocks = nn.ModuleList()

        self.in_proj = nn.ModuleList()
        self.hire_fusion = nn.ModuleList()

        for i in range(num_layers):
            layer = VSSLayer(dim=dims[i],
                             depth=depths[i],
                             downsample=UpSample2D,
                             dim_out=dims[i + 1] if i != num_layers - 1 else dims[i] // 2,
                             forward_coremm='SS2D')
            self.multimodal_blocks.append(layer)

            self.in_proj.append(Linear2d(3*dims[i], dims[i], bias=False))

            self.text_guidance.append(nn.Sequential(nn.Linear(768, dims[i]), nn.ReLU()))

            self.local_text_fusion.append(ImageTextCorr(visual_dim=dims[i],
                                                        text_dim=768,
                                                        hidden_dim=embedding_dim,
                                                        out_dim=dims[i]))

            if i != num_layers - 1:
                self.hire_fusion.append(Fusion(dims[i + 1]))

        self.proj_out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(dims[3] // 2, dims[3] // 2, 3, padding=1),
            nn.Conv2d(dims[3] // 2, num_classes, 3, padding=1),
        )


    def forward(self, x, l_feat, l_mask):
        x = x[::-1]
        feat = x[0]

        for i, layer in enumerate(self.multimodal_blocks):
            _, c, h, w = feat.shape

            pooling_text = l_feat[..., 0]

            text_guidance = self.text_guidance[i](pooling_text)
            text_guidance = einops.repeat(text_guidance, "b c -> b c h w", h=h, w=w)

            local_text = self.local_text_fusion[i](feat, l_feat, l_mask)
            local_text = einops.rearrange(local_text, 'b h w c -> b c h w', h=h)

            mm_input = torch.cat([feat, text_guidance, local_text], dim=1)
            feat, _ = layer(self.in_proj[i](mm_input))

            if i + 1 < len(x):
                feat = self.hire_fusion[i](feat, x[i + 1])
                feat = feat + x[i + 1]

        out = self.proj_out(feat)

        return out
