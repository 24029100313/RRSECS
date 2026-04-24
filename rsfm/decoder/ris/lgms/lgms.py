from torch import nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from .utils import GuideDecoder
from einops import rearrange



class LGMS(nn.Module):
    def __init__(self,
                 img_size=224,
                 in_channels=[32, 64, 160, 256],
                 embedding_dim=64,
                 num_classes=2,
                 **kwargs):
        super(LGMS, self).__init__()
        feature_dim = in_channels[::-1]
        self.spatial_dim = [(img_size // 32) * 2 ** i for i in range(4)]

        self.decoder16 = GuideDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 20)
        self.decoder8 = GuideDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 20)
        self.decoder4 = GuideDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 20)
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=num_classes)

    def forward(self, inputs, l):
        xs = [rearrange(x, 'b c h w -> b (h w) c') for x in inputs]
        x1, x2, x3, x4 = xs

        x3 = self.decoder16(x4, x3, l)
        x2 = self.decoder8(x3, x2, l)
        x1 = self.decoder4(x2, x1, l)
        x1 = rearrange(x1, 'B (H W) C -> B C H W', H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        x1 = self.decoder1(x1)

        out = self.out(x1)

        return out