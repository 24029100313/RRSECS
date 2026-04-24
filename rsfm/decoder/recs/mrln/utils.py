import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class FFRL(nn.Module):
    """
    Feature-Feature Relational Learning 模块
    论文核心组件，负责学习视觉特征之间的空间关系
    Args:
        in_channels (int): 输入特征图的通道数
        h_dim (int): 隐藏层维度，默认1024
    """
    def __init__(self, v_dim, l_dim, h_dim=256):
        super().__init__()

        # ----------------- 特征融合部分 -----------------
        self.W_v = nn.Linear(v_dim, v_dim)
        self.W_t = nn.Linear(l_dim, v_dim)
        self.prelu_W_v = nn.PReLU()
        self.prelu_W_t = nn.PReLU()

        # ----------------- 关系学习部分 -----------------
        # Pair-wise关系流 (1x1卷积)
        self.pair_stream = nn.Sequential(
            nn.Conv2d(v_dim, h_dim, 1),
            nn.PReLU(),
            nn.Conv2d(h_dim, 1, 1),
            nn.PReLU()
        )
        
        # Inner-group关系流 (3x3卷积)
        self.group_stream = nn.Sequential(
            nn.Conv2d(v_dim, h_dim, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(h_dim, 1, 3, padding=1),
            nn.PReLU()
        )

    def forward(self, F_v, f_t):
        """
        前向传播
        Args:
            F_v (Tensor): 视觉特征图 [B, C, H, W]
            f_t (Tensor): 文本特征向量 [B, h_dim]
        Returns:
            Tensor: 关系增强后的特征图 [B, C, H, W]
        """
        B, C, H, W = F_v.shape
        
        # ============= 特征融合 (公式1) =============
        visual_proj = self.prelu_W_v(self.W_v(F_v.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))  # [B,C,H,W]
        text_proj = self.prelu_W_t(self.W_t(f_t))    # [B, C]
        text_proj = text_proj.view(B, C, 1, 1)   # [B, C, 1, 1]
        
        # Hadamard积融合
        F_s = visual_proj * text_proj            # [B, C, H, W]

        # ============= 关系矩阵计算 =============
        # 展平空间维度
        F_s_flat = F_s.view(B, C, H*W).permute(0, 2, 1)  # [B, K, C], K=H*W
        text_proj = text_proj.view(B, 1, C)
        F_s_flat = F_s_flat + text_proj
        
        # 构造对象对矩阵
        F_sr = F_s_flat.unsqueeze(2).expand(-1, -1, H*W, -1)  # [B, K, K, C]
        V_pw = F_sr + F_sr.permute(0, 2, 1, 3)  # 对称加法 [B, K, K, C]

        # ----------------- Pair-wise关系 -----------------
        V_pw = V_pw.permute(0, 3, 1, 2)          # [B, C, K, K]
        V_p = self.pair_stream(V_pw).squeeze(1)             # [B, K, K]
        R_p = F.softmax(V_p + V_p.permute(0, 2, 1), dim=-1)  # [B, K, K]

        # ----------------- Inner-group关系 -----------------
        V_g = self.group_stream(V_pw).squeeze(1)             # [B, K, K]
        R_g = F.softmax(V_g + V_g.permute(0, 2, 1), dim=-1)  # [B, K, K]

        # ============= 特征更新 (公式5) =============
        F_v_flat = F_v.view(B, C, H*W)            # [B, C, K]
        updated_feat = torch.einsum('bkk,bck->bck', (R_p + R_g), F_v_flat)  # [B, C, K]
        
        return updated_feat.view(B, C, H, W)      # 恢复空间维度


class FTRL(nn.Module):
    """ 修正维度映射错误的Feature-Task关系学习模块 """
    def __init__(self, v_dim, l_dim):
        super().__init__()
        # 视觉特征投影层（输入维度需匹配视觉特征通道数）
        self.W_va = nn.Linear(v_dim, l_dim)  # 关键修正：输入维度=visual_dim
        self.prelu_W_va = nn.PReLU()
        # 文本特征投影层
        self.W_ta = nn.Linear(l_dim, l_dim)
        self.prelu_W_ta = nn.PReLU()

    def forward(self, F_m, f_t, task_emb=None):
        """
        Args:
            F_m (Tensor): 视觉特征 [B, C, ...]
            f_t (Tensor): 文本特征 [B, text_dim]
        """
        B, C, H, W = F_m.shape
        
        # 步骤1：视觉特征投影 (修正维度)
        # [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
        visual_flat = F_m.permute(0,2,3,1).contiguous().view(-1, C)
        # 投影到文本维度 [B*H*W, text_dim]
        visual_proj = self.prelu_W_va(self.W_va(visual_flat))  
        # 恢复形状 [B, H, W, text_dim]
        visual_proj = visual_proj.view(B, H, W, -1)
        
        # 步骤2：文本特征投影 [B, text_dim] -> [B, 1, 1, text_dim]
        text_proj = self.prelu_W_ta(self.W_ta(f_t)).unsqueeze(1).unsqueeze(1)
        
        # 步骤3：注意力计算
        e_c = torch.matmul(visual_proj, text_proj.transpose(-1, -2))  # [B, H, W, 1]
        A_col = F.softmax(e_c.squeeze(-1), dim=-1)  # [B, H, W]
        
        # Task attention (Eq.8-9)
        if task_emb is not None:
            s_t = F_m + task_emb.unsqueeze(-1).unsqueeze(-1)
            task_proj = F.prelu(self.W_va(s_t.permute(0,2,3,1)))
            e_t = torch.matmul(task_proj, text_proj.transpose(-1,-2)).squeeze(-1)
            A_task = torch.sigmoid(e_t)
        else:
            A_task = torch.ones_like(A_col)
            
        # Feature fusion (Eq.10)
        f_att = torch.einsum('bcij,bij->bc', F_m, A_col)
        F_att = torch.einsum('bc,bij->bcij', f_att, A_task)
        # F_att = f_att.view_as(F_m)
        return F_m + F_att


class TTRL(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        
    def gaussian_kernel(self, x, y):
        x_sqnorms = torch.sum(x**2, dim=-1, keepdim=True)
        y_sqnorms = torch.sum(y**2, dim=-1, keepdim=True)
        pairwise_dists = x_sqnorms - 2*torch.matmul(x, y.t()) + y_sqnorms.t()
        return torch.exp(-pairwise_dists / (2 * self.sigma**2))
    
    def forward(self, F_s, F_c):
        # F_s: RES features [B, H*W, D]
        # F_c: REC features [B, H*W, D]
        kernel_ss = self.gaussian_kernel(F_s, F_s).mean()
        kernel_cc = self.gaussian_kernel(F_c, F_c).mean()
        kernel_sc = self.gaussian_kernel(F_s, F_c).mean()
        mmd_loss = kernel_ss + kernel_cc - 2*kernel_sc
        return mmd_loss


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


class aspp_decoder(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Layer

    Args:
        planes (int): input channels
        hidden_planes (int): middle channels
        out_planes (int): output channels
    """
    def __init__(self, planes, hidden_planes, out_planes):
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


class MCNhead(nn.Module):
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
        n_classes=0, 
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

        super(MCNhead, self).__init__()
        self.anchors = anchors
        self.anch_mask = arch_mask[layer_no]
        self.n_anchors = len(self.anch_mask)
        self.n_classes = n_classes
        self.stride = 32 # strides[layer_no]
        self.all_anchors_grid = [(w / self.stride, h / self.stride)
                                 for w, h in self.anchors]
        self.masked_anchors = [self.all_anchors_grid[i]
                               for i in self.anch_mask]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)

        self.d_proj = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)
        self.s_proj = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)

        self.dconv = nn.Conv2d(in_channels=in_ch,
                              out_channels=self.n_anchors * (5), # xywh,conf
                              kernel_size=1, stride=1, padding=0)
        self.sconv=nn.Sequential(aspp_decoder(in_ch, hidden_size//2, self.n_classes),
                                 nn.UpsamplingBilinear2d(scale_factor=8)
                                )
    
    def forward(self, xin, yin):
        _, _, f_h, f_w = xin.shape

        output = self.dconv(xin)
        mask=self.sconv(yin)

        batchsize = output.shape[0]
        fsize = output.shape[2]
        n_ch = 5
        dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor
        devices=xin.device

        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize) # H/32, W/32
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

        pred = output.clone()
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors

        pred[..., :4] /= f_h
        pred = pred.view(batchsize, -1, n_ch)
        score = pred[:,:,4].sigmoid()
        max_score, ind = torch.max(score, -1)
        ind = ind.unsqueeze(1).unsqueeze(1).repeat(1, 1, n_ch)
        pred = torch.gather(pred, 1, ind)

        return pred.view(batchsize,-1)[:, :4], mask


class MultiScaleFusion(nn.Module):
    def __init__(self,v_planes=[256,512,1024], hiden_planes=512, scaled=True):
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

    def forward(self, x):
        l,m,s=x
        m = torch.cat([self.up_modules[1](s), m], 1)
        l = torch.cat([self.up_modules[0](m), l], 1)
        # out=self.out_proj(l)

        m = torch.cat([self.down_modules[0](l), m], 1)

        s = torch.cat([self.down_modules[1](m), s], 1)

        #top prpj and bot proj
        top_feat=self.top_proj(s)
        mid_feat=self.mid_proj(m)
        bot_feat=self.bot_proj(l)
        return [bot_feat,mid_feat,top_feat]