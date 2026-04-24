import torch
from torch import nn
from .utils import FFRL, FTRL, MCNhead



class MRLNHead(nn.Module):
    def __init__(self, 
                 in_channels=[96, 192, 384, 768], 
                 embedding_dim=512, 
                 num_classes=2,
                 l_dim=768,
                 **kwargs):
        super(MRLNHead, self).__init__()
        
        # 多关系学习模块
        self.ffrl = FFRL(v_dim=in_channels[-1], l_dim=l_dim, h_dim=embedding_dim)

        # 上采样层
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # FTRL模块初始化
        self.ftrl_res = FTRL(v_dim=in_channels[-1], l_dim=l_dim)
        self.ftrl_rec = FTRL(v_dim=in_channels[-1], l_dim=l_dim)
        
        # 任务头
        self.head = MCNhead(hidden_size=embedding_dim, anchors=[[137, 256], [248, 272], [386, 271]],
                            arch_mask=[[0, 1, 2]], layer_no=0, in_ch=in_channels[-1], n_classes=num_classes)
        
    def forward(self, feats, lang, lang_mask=None):
        """
        前向传播
        Args:
            feats (Tensor): 视觉特征图 [B, v_dim, H, W]
            lang (Tensor): 文本特征向量 [B, N, l_dim]
        Returns:
            dict: 
        """
        # 特征提取
        x1, x2, x3, x4 = feats
        g_lang = torch.mean(lang.permute(0, 2, 1), dim=1, keepdim=False) # [B, l_dim]
        
        # FFRL处理
        f_m1 = self.ffrl(x4, g_lang) # [B, 768, 16, 16]
        f_m2 = self.upsample1(f_m1) # [B, 768, 32, 32]
        f_m3 = self.upsample2(f_m2) # [B, 768, 64, 64]
        
        # FTRL
        res_feat = self.ftrl_res(f_m3, g_lang) # [B, 768, 64, 64]
        rec_feat = self.ftrl_rec(f_m1, g_lang) # [B, 768, 16, 16]

        pred_bbox, pred_mask = self.head(rec_feat, res_feat)
        
        return {'pred_bboxs': pred_bbox,
                'pred_masks': pred_mask}