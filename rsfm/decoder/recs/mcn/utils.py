import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


class MCN_fusion(nn.Module):
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 l_dim=768,
                 hidden_dim=512,
                 num_heads=2,
                 fusion_drop=0.1,
                 scaled_multiscale=True,
                 **kwargs):
        super().__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.fusion_manner=SimpleFusion(
            v_planes=c4_in_channels,
            q_planes=l_dim,
            out_planes=c4_in_channels,
        )

        self.multi_scale_manner=MultiScaleFusion(
            v_planes= (c2_in_channels, c3_in_channels, c4_in_channels),
            hiden_planes=hidden_dim,
            scaled=scaled_multiscale
        )

        # GaranAttention
        self.seg_garan=GaranAttention(
            d_q=l_dim,
            d_v=hidden_dim,
            n_head=num_heads,
            dropout=fusion_drop
        )

        self.det_garan=GaranAttention(
            d_q=l_dim,
            d_v=hidden_dim,
            n_head=num_heads,
            dropout=fusion_drop
        )

    def forward(self,x, y):
        _, x_large, x_middle, x_small = x

        # SimpleFusion
        flat_lang_feat = torch.mean(y, dim=2)# (B, l_dim, N_l)  -> (B, D_t)
        x_small = self.fusion_manner(x_small ,flat_lang_feat)

        bot_feats, _, top_feats=self.multi_scale_manner(x_large, x_middle, x_small)
        # RES branch
        bot_feats,seg_map,seg_attn=self.seg_garan(flat_lang_feat, bot_feats)
        # REC branch
        top_feats,det_map,det_attn=self.det_garan(flat_lang_feat, top_feats)

        return det_attn, seg_attn


class MCN_decoder(nn.Module):
    """
    detection layer corresponding to yolo_layer.c of darknet
    """

    def __init__(
            self,
            hidden_size=512,
            anchors=[[137, 256], [248, 272], [386, 271]],
            arch_mask=[[0, 1, 2]],
            layer_no=0,
            in_ch=512,
            n_classes=2,
    ):
        """
        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
        """

        super(MCN_decoder, self).__init__()
        self.anchors = anchors
        self.anch_mask = arch_mask[layer_no]
        self.n_anchors = len(self.anch_mask)
        self.n_classes = n_classes
        self.stride = 32  # strides[layer_no]
        self.all_anchors_grid = [(w / self.stride, h / self.stride)
                                 for w, h in self.anchors]
        self.masked_anchors = [self.all_anchors_grid[i]
                               for i in self.anch_mask]
        ref_anchors = torch.zeros(len(self.all_anchors_grid), 4)
        ref_anchors[:, 2:] = torch.tensor(self.all_anchors_grid)
        # self.ref_anchors = ref_anchors
        self.register_buffer("ref_anchors", ref_anchors)

        self.d_proj = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)
        self.s_proj = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)

        self.dconv = nn.Conv2d(in_channels=in_ch,
                               out_channels=self.n_anchors * (5),  # xywh,conf
                               kernel_size=1, stride=1, padding=0)
        self.sconv = nn.Sequential(aspp_decoder(in_ch, hidden_size // 2, self.n_classes),
                                   nn.UpsamplingBilinear2d(scale_factor=8)
                                   )

    def forward(self, det_feat, seg_attn):
        _, _, f_h, f_w = det_feat.shape

        output = self.dconv(det_feat)
        mask = self.sconv(seg_attn)

        batchsize = output.shape[0]
        fsize = output.shape[2]
        n_ch = 5
        dtype = torch.cuda.FloatTensor if det_feat.is_cuda else torch.FloatTensor
        devices = det_feat.device

        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2).contiguous()

        # logistic activation for xy, obj, cls
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
            output[..., np.r_[:2, 4:n_ch]])

        # calculate pred - xywh obj cls

        x_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32), output.shape[:4])).to(devices)
        y_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4])).to(devices)

        masked_anchors = np.array(self.masked_anchors)

        w_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 0], (1, self.n_anchors, 1, 1)), output.shape[:4])).to(devices)
        h_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 1], (1, self.n_anchors, 1, 1)), output.shape[:4])).to(devices)

        pred = output
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors

        pred[..., :4] /= f_h
        pred = pred.view(batchsize, -1, n_ch)
        score = pred[:, :, 4].sigmoid()
        max_score, ind = torch.max(score, -1)
        ind = ind.unsqueeze(1).unsqueeze(1).repeat(1, 1, n_ch)
        pred = torch.gather(pred, 1, ind)

        pred_bbox = pred.view(batchsize, -1)[:, :4]
        pred_mask = mask

        return {'pred_bboxs': pred_bbox, 'pred_masks': pred_mask}


def darknet_conv(in_ch, out_ch, ksize, stride=1, dilation_rate=1):
    """
    Add a darknet-style convolution block as Conv-Bn-LeakyReLU.
    
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
        dilation_rate (int): spacing between kernel elements.
    
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (dilation_rate * (ksize - 1) + 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False,dilation=dilation_rate))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('leaky', nn.LeakyReLU(0.1))
    return stage


class SimpleFusion(nn.Module):
    def __init__(self,v_planes=1024,q_planes=1024,out_planes=1024):
        super().__init__()
        self.v_proj=nn.Sequential(
            nn.Conv2d(v_planes, out_planes, 1),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
        self.q_proj=nn.Sequential(
            nn.Conv2d(q_planes, out_planes, 1),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
        self.norm=nn.Sequential(
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x, y):
        x=self.v_proj(x)
        y=self.q_proj(y.unsqueeze(2).unsqueeze(2))
        return self.norm(x*y)


class MultiScaleFusion(nn.Module):
    def __init__(self,v_planes=[256, 512, 1024], hiden_planes=512, scaled=True):
        super().__init__()
        self.up_modules=nn.ModuleList(
            [nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
                darknet_conv(v_planes[-2]+hiden_planes//2, hiden_planes//2, ksize=1),
                darknet_conv(hiden_planes//2, hiden_planes//2, 3),
            ),
            nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
                darknet_conv(v_planes[-1], hiden_planes//2, ksize=1),
                darknet_conv(hiden_planes//2, hiden_planes//2, 3)
            )]
        )

        self.down_modules=nn.ModuleList(
            [nn.Sequential(
                nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
                darknet_conv(hiden_planes//2+v_planes[0], hiden_planes // 2, ksize=1),
                darknet_conv(hiden_planes // 2, hiden_planes//2, 3),
            ),
                nn.Sequential(
                nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
                darknet_conv(hiden_planes+v_planes[1], hiden_planes//2, ksize=1),
                darknet_conv(hiden_planes//2, hiden_planes//2, 3),
            )]
        )


        self.top_proj=darknet_conv(v_planes[-1]+hiden_planes//2,hiden_planes,1)
        self.mid_proj=darknet_conv(v_planes[1]+hiden_planes,hiden_planes,1)
        self.bot_proj=darknet_conv(v_planes[0]+hiden_planes//2,hiden_planes,1)

    def forward(self, x_large, x_middle, x_small):
        x_middle = torch.cat([self.up_modules[1](x_small), x_middle], 1)
        x_large = torch.cat([self.up_modules[0](x_middle), x_large], 1)

        x_middle = torch.cat([self.down_modules[0](x_large), x_middle], 1)

        x_small = torch.cat([self.down_modules[1](x_middle), x_small], 1)

        #top prpj and bot proj
        top_feat=self.top_proj(x_small)
        mid_feat=self.mid_proj(x_middle)
        bot_feat=self.bot_proj(x_large)
        return [bot_feat, mid_feat, top_feat]


class CollectDiffuseAttention(nn.Module):
    ''' CollectDiffuseAttention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout_c = nn.Dropout(attn_dropout)
        self.dropout_d = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, q, kc,kd, v, mask=None):
        '''
        q: n*b,1,d_o
        kc: n*b,h*w,d_o
        kd: n*b,h*w,d_o
        v: n*b,h*w,d_o
        '''

        attn_col = torch.bmm(q, kc.transpose(1, 2)) #n*b,1,h*w
        attn_col_logit = attn_col / self.temperature
        attn_col = self.softmax(attn_col_logit)
        attn_col = self.dropout_c(attn_col)
        attn = torch.bmm(attn_col, v) #n*b,1,d_o

        attn_dif = torch.bmm(kd,q.transpose(1, 2)) #n*b,h*w,1
        attn_dif_logit = attn_dif / self.temperature
        attn_dif = torch.sigmoid(attn_dif_logit)
        attn_dif= self.dropout_d(attn_dif)
        output=torch.bmm(attn_dif,attn)
        return output, attn_col_logit.squeeze(1)


class GaranAttention(nn.Module):
    """
    Garan Attention Module
    """

    def __init__(self,d_q, d_v,n_head=2, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_q = d_q
        self.d_v = d_v
        self.d_k = d_v
        self.d_o = d_v
        d_o = d_v

        self.w_qs = nn.Linear(d_q, d_o)
        self.w_kc = nn.Conv2d(d_v, d_o, 1)
        self.w_kd = nn.Conv2d(d_v, d_o, 1)
        self.w_vs = nn.Conv2d(d_v, d_o, 1)
        self.w_m = nn.Conv2d(d_o, 1, 3, 1,padding=1)
        self.w_o = nn.Conv2d(d_o, d_o, 1)

        self.attention = CollectDiffuseAttention(temperature=np.power(d_o//n_head, 0.5))
        self.layer_norm = nn.BatchNorm2d(d_o)
        self.layer_acti= nn.LeakyReLU(0.1,inplace=True)
        # nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, v, mask=None):

        d_k, d_v, n_head, d_o = self.d_k, self.d_v, self.n_head, self.d_o

        sz_b, c_q = q.size()
        sz_b,c_v, h_v,w_v = v.size()
        residual = v

        q = self.w_qs(q)
        kc=self.w_kc(v).view(sz_b,n_head,d_o//n_head,h_v*w_v)
        kd=self.w_kd(v).view(sz_b,n_head,d_o//n_head,h_v*w_v)
        v=self.w_vs(v).view(sz_b,n_head,d_o//n_head,h_v*w_v)
        q=q.view(sz_b,n_head,1,d_o//n_head)
        # v=v.view(sz_b,h_v*w_v,n_head,c_v//n_head)

        q = q.view(-1, 1, d_o//n_head) # (n*b) x lq x dk
        kc = kc.permute(0,1,3,2).contiguous().view(-1, h_v*w_v, d_o//n_head) # (n*b) x lk x dk
        kd=kd.permute(0,1,3,2).contiguous().view(-1, h_v*w_v, d_o//n_head) # (n*b) x lk x dk
        v = v.permute(0,1,3,2).contiguous().view(-1, h_v*w_v, d_o//n_head) # (n*b) x lv x dv

        output, m_attn = self.attention(q, kc,kd, v)
        #n * b, h * w, d_o
        output = output.view(sz_b,n_head, h_v,w_v, d_o//n_head)
        output = output.permute(0,1,4,3,2).contiguous().view(sz_b,-1, h_v,w_v) # b x lq x (n*dv)
        m_attn=m_attn.view(sz_b,n_head, h_v*w_v)
        # m_attn=m_attn.mean(1)

        #residual connect
        output=self.w_o(output)
        attn=output
        m_attn=self.w_m(attn).view(sz_b, h_v*w_v)
        output=self.layer_norm(output)
        output= output+residual
        output=self.layer_acti(output)

        # output = self.dropout(self.fc(output))

        return output, m_attn, attn


def co_energe(self,x_map,y_map,x_attn,y_attn,eps=1e-6):
    """
    :param x_map:  h*w
    :param y_map: h*w
    :param x_attn: B,c,h,w
    :param y_attn: B,c,h,w
    :return:
    """
    b,c,h,w=x_attn.size()
    x_map=F.softmax(self.s_proj(x_attn).view(b,-1),-1)
    y_map=F.softmax(self.d_proj(y_attn).view(b,-1),-1)
    x_attn=F.normalize(x_attn,dim=1).view(b,c,-1)
    y_attn=F.normalize(y_attn,dim=1).view(b,c,-1)
    cosin_sim=torch.bmm(x_attn.transpose(1,2),y_attn) #b,h_x*w_x,h_y*w_y
    cosin_sim=(cosin_sim+1.)/2.
    co_en=torch.einsum('blk,bl,bk->b',[cosin_sim,x_map,y_map])
    return -torch.log(co_en+eps)


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


class aspp_decoder(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Layer

    Args:
        planes (int): input channels
        hidden_planes (int): middle channels
        out_planes (int): output channels
    """
    def __init__(self, planes,hidden_planes,out_planes):
        super().__init__()
        self.conv0 = darknet_conv(planes, hidden_planes, ksize=1, stride=1)
        self.conv1 = darknet_conv(planes, hidden_planes, ksize=3, stride=1,dilation_rate=6)
        self.conv2 = darknet_conv(planes, hidden_planes, ksize=3, stride=1,dilation_rate=12)
        self.conv3 = darknet_conv(planes, hidden_planes, ksize=3, stride=1,dilation_rate=18)
        self.conv4 = darknet_conv(planes, hidden_planes, ksize=1, stride=1)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.out_proj= nn.Conv2d(hidden_planes*5, out_planes, 1)
    def forward(self, x):
        _, _, h, w = x.size()
        b0 = self.conv0(x)
        b1 = self.conv1(x)
        b2 = self.conv2(x)
        b3 = self.conv3(x)
        b4 = self.conv4(self.pool(x)).repeat(1,1,h,w)
        x=torch.cat([b0,b1,b2,b3,b4],1)
        x=self.out_proj(x)
        return x

