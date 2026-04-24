import torch.nn as nn
from .utils import MSCAM, EUCB, LGAG



class EMCADHead(nn.Module):
    '''
    from EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation, CVPR 2024
    '''
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 embedding_dim=256,
                 num_classes=1,
                 kernel_sizes=[1, 3, 5],
                 expansion_factor=2,
                 dw_parallel=True,
                 add=True,
                 activation='relu',
                 aux_loss=False,
                 **kwargs):
        super(EMCADHead, self).__init__()

        self.aux_loss = aux_loss

        self.mscam4 = MSCAM(in_channels=in_channels[3], out_channels=in_channels[3], kernel_sizes=kernel_sizes,
                            expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

        self.eucb3 = EUCB(in_channels=in_channels[3], out_channels=in_channels[2],
                          kernel_size=3, stride=1, activation=activation)
        self.lgag3 = LGAG(F_g=in_channels[2], F_l=in_channels[2], F_int=in_channels[2] // 2,
                          kernel_size=3, groups=in_channels[2] // 2, activation=activation)
        self.mscam3 = MSCAM(in_channels=in_channels[2], out_channels=in_channels[2], kernel_sizes=kernel_sizes,
                            expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

        self.eucb2 = EUCB(in_channels=in_channels[2], out_channels=in_channels[1],
                          kernel_size=3, stride=1, activation=activation)
        self.lgag2 = LGAG(F_g=in_channels[1], F_l=in_channels[1], F_int=in_channels[1] // 2,
                          kernel_size=3, groups=in_channels[1] // 2, activation=activation)
        self.mscam2 = MSCAM(in_channels=in_channels[1], out_channels=in_channels[1], kernel_sizes=kernel_sizes,
                            expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

        self.eucb1 = EUCB(in_channels=in_channels[1], out_channels=in_channels[0],
                          kernel_size=3, stride=1, activation=activation)
        self.lgag1 = LGAG(F_g=in_channels[0], F_l=in_channels[0], F_int=in_channels[0] // 2,
                          kernel_size=3, groups=in_channels[0] // 2, activation=activation)
        self.mscam1 = MSCAM(in_channels=in_channels[0], out_channels=in_channels[0], kernel_sizes=kernel_sizes,
                            expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

        if self.aux_loss:
            self.out_head4 = nn.Conv2d(in_channels[3], num_classes, 1)
            self.out_head3 = nn.Conv2d(in_channels[2], num_classes, 1)
            self.out_head2 = nn.Conv2d(in_channels[1], num_classes, 1)
            self.out_head1 = nn.Conv2d(in_channels[0], num_classes, 1)
        else:
            self.out_head = nn.Conv2d(in_channels[0], num_classes, 1)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


    def forward(self, inputs):
        enc_x1, enc_x2, enc_x3, enc_x4 = inputs

        # MSCAM4
        d4 = self.mscam4(enc_x4)

        # EUCB3
        d3 = self.eucb3(d4)
        # LGAG3
        x3 = self.lgag3(g=d3, x=enc_x3)
        # Additive aggregation 3
        d3 = d3 + x3
        # MSCAM3
        d3 = self.mscam3(d3)

        # EUCB2
        d2 = self.eucb2(d3)
        # LGAG2
        x2 = self.lgag2(g=d2, x=enc_x2)
        # Additive aggregation 2
        d2 = d2 + x2
        # MSCAM2
        d2 = self.mscam2(d2)

        # EUCB1
        d1 = self.eucb1(d2)
        # LGAG1
        x1 = self.lgag1(g=d1, x=enc_x1)
        # Additive aggregation 1
        d1 = d1 + x1
        # MSCAM1
        d1 = self.mscam1(d1)

        if self.aux_loss:
            p4 = self.out_head4(d4)
            p3 = self.out_head3(d3)
            p2 = self.out_head2(d2)
            p1 = self.out_head1(d1)

            if self.training:
                return [p4, p3, p2, p1]
            else:
                return p1
        else:
            p = self.out_head(d1)

            return p