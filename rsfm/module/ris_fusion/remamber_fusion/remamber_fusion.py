import einops, math
from torch import nn
from .utils import VSSLayer, ImageTextCorr



class ReMamber_fusion(nn.Module):
    def __init__(self, dim, l_dim):
        super().__init__()

        self.text_guidance = nn.Sequential(nn.Linear(l_dim, dim),
                                           nn.ReLU())

        self.local_text_fusion = ImageTextCorr(visual_dim=dim,
                                               text_dim=l_dim,
                                               hidden_dim=512,
                                               out_dim=dim)

        self.multimodal_block = VSSLayer(dim)

    def forward(self, x, l, l_mask):
        _, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.view(-1, h, w, c).permute(0, 3, 1, 2)

        pooling_text = l[..., 0]

        text_guidance = self.text_guidance(pooling_text)
        text_guidance = einops.repeat(text_guidance, "b c -> b c h w", h=h, w=w)

        local_text = self.local_text_fusion(x, l, l_mask)
        local_text = einops.rearrange(local_text, 'b h w c -> b c h w', h=h)

        mm_input = (x, text_guidance, local_text)
        out, out_residual = self.multimodal_block(mm_input)

        res = out[0].flatten(2).transpose(1, 2)
        res_residual = out_residual[0].flatten(2).transpose(1, 2)

        return res, res_residual