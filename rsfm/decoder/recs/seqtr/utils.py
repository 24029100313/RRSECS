import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.utils import build_transformer
from mmdet.models.utils import build_linear_layer
from mmcv.cnn import build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout
from .transformer import AutoRegressiveTransformer


def darknet_conv(in_chs,
                 out_chs,
                 kernel_sizes,
                 strides,
                 norm_cfg=dict(type="BN2d"),
                 act_cfg=dict(type="LeakyReLU", negative_slope=0.1)):
    convs = []
    for i, (in_ch, out_ch, kernel_size, stride) in enumerate(zip(in_chs, out_chs, kernel_sizes, strides)):
        convs.append(ConvModule(in_ch,
                                out_ch,
                                kernel_size,
                                stride=stride,
                                padding=kernel_size // 2,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg))

    return convs


class SimpleFusion(nn.Module):
    def __init__(self,
                 vis_chs=(256, 512, 1024),
                 direction='bottom_up',
                 l_dim=768):
        super(SimpleFusion, self).__init__()
        self.fp16_enabled = False
        assert direction in ['to_mid', 'bottom_up', 'none']
        self.direction = direction

        if direction == 'bottom_up':
            assert len(vis_chs) == 3
            ch = sum(vis_chs[:2])
            self.down_mid2top = nn.Sequential(
                nn.AvgPool2d(2, 2),
                *darknet_conv((ch, ), (ch, ), (3, ), (1, )))
            self.down_bot2mid = nn.Sequential(
                nn.AvgPool2d(2, 2),
                *darknet_conv((vis_chs[0], ), (vis_chs[0], ), (3, ), (1, )))
            self.top_project = nn.Sequential(
                *darknet_conv((ch+vis_chs[-1], ch+vis_chs[-1], ), (ch+vis_chs[-1], vis_chs[-1], ), (3, 1, ), (1, 1, )))
        elif direction == 'to_mid':
            assert len(vis_chs) == 3
            ch = sum(vis_chs)
            self.up_top2mid = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                *darknet_conv((vis_chs[-1], ), (vis_chs[-1], ), (3, ), (1, )))
            self.down_bot2mid = nn.Sequential(
                nn.AvgPool2d(2, 2),
                *darknet_conv((vis_chs[0], ), (vis_chs[0], ), (3, ), (1, )))
            self.mid_project = nn.Sequential(
                *darknet_conv((ch, ch, ), (ch, vis_chs[-1], ), (3, 1, ), (3, 1, )))
        elif direction == 'none':
            assert len(vis_chs) == 1

        self.activate = nn.Tanh()
        self.lang_proj = nn.Linear(l_dim, vis_chs[-1])

    def forward(self, x, y):
        if self.direction == 'bottom_up':
            l, m, s = x
            m = torch.cat([self.down_bot2mid(l), m], 1)
            s = torch.cat([self.down_mid2top(m), s], 1)
            x_vis_enc = self.top_project(s)
        elif self.direction == 'to_mid':
            l, m, s = x
            s = self.up_top2mid(s)
            l = self.down_bot2mid(l)
            m = torch.cat([s, m, l], 1)
            x_vis_enc = self.mid_project(m)
        elif self.direction == 'none':
            x_vis_enc = x

        y_2d = self.lang_proj(y).unsqueeze(-1).unsqueeze(-1)
        x_multi_modal = self.activate(x_vis_enc) * self.activate(y_2d)

        return x_multi_modal


class LinearModule(nn.Module):
    """A linear block that bundles linear/activation/dropout layers.

    This block simplifies the usage of linear layers, which are commonly
    used with an activation layer (e.g., ReLU) and Dropout layer (e.g., Dropout).
    It is based upon three build methods: `build_linear_layer()`,
    `build_activation_layer()` and `build_dropout`.

    Args:
        linear (dict): Config dict for activation layer. Default: dict(type='Linear', bias=True)
        act (dict): Config dict for activation layer. Default: dict(type='ReLU', inplace=True).
        drop (dict): Config dict for dropout layer. Default: dict(type='Dropout', drop_prob=0.5)
    """

    def __init__(self,
                 linear=dict(type='Linear', bias=True),
                 act=dict(type='ReLU', inplace=True),
                 drop=dict(type='Dropout', drop_prob=0.5)):
        super(LinearModule, self).__init__()
        assert linear is None or isinstance(linear, dict)
        assert act is None or isinstance(act, dict)
        assert drop is None or isinstance(drop, dict)
        assert 'in_features' in linear and 'out_features' in linear

        self.with_activation = act is not None
        self.with_drop = drop is not None

        self.fc = build_linear_layer(linear)

        if self.with_activation:
            self.activate = build_activation_layer(act)

        if self.with_drop:
            self.drop = build_dropout(drop)

    def forward(self, input):
        input = self.fc(input)

        if self.with_activation:
            input = self.activate(input)

        if self.with_drop:
            input = self.drop(input)

        return input


class SeqHead(nn.Module):
    def __init__(self,
                 in_ch=1024,
                 num_bin=1000,
                 multi_task=True,
                 shuffle_fraction=-1,
                 mapping="relative",
                 top_p=-1,
                 num_ray=18,
                 det_coord=[0],
                 det_coord_weight=1.5,
                 predictor=dict(
                     num_fcs=3, in_chs=[256, 256, 256], out_chs=[256, 256, 1001],
                     fc=[
                         dict(
                             linear=dict(type='Linear', bias=True),
                             act=dict(type='ReLU', inplace=True),
                             drop=None),
                         dict(
                             linear=dict(type='Linear', bias=True),
                             act=dict(type='ReLU', inplace=True),
                             drop=None),
                         dict(
                             linear=dict(type='Linear', bias=True),
                             act=None,
                             drop=None)
                     ]
                 ),
                 transformer=dict(
                     encoder=dict(
                         num_layers=6,
                         layer=dict(
                             d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu',
                             batch_first=True)),
                     decoder=dict(
                         num_layers=3,
                         layer=dict(
                             d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu',
                             batch_first=True),
                     )),
                 x_positional_encoding=dict(
                     type='SinePositionalEncoding2D',
                     num_feature=128,
                     normalize=True),
                 seq_positional_encoding=dict(
                     type='LearnedPositionalEncoding1D',
                     num_embedding=5,
                     # num_embedding=42,
                     num_feature=256)
                 ):
        super(SeqHead, self).__init__()
        self.num_bin = num_bin
        self.multi_task = multi_task
        self.shuffle_fraction = shuffle_fraction
        assert mapping in ["relative", "absolute"]
        self.mapping = mapping
        self.top_p = top_p
        self.num_ray = num_ray
        self.det_coord = det_coord
        self.det_coord_weight = det_coord_weight

        self.transformer = AutoRegressiveTransformer(transformer['encoder'],
                                                     transformer['decoder'])
        self.d_model = self.transformer.d_model

        self._init_layers(in_ch,
                          predictor,
                          multi_task,
                          x_positional_encoding,
                          seq_positional_encoding)

    def _init_layers(self,
                     in_ch,
                     predictor_cfg,
                     multi_task,
                     x_positional_encoding,
                     seq_positional_encoding):
        num_fcs = predictor_cfg.pop('num_fcs')
        in_chs, out_chs = predictor_cfg.pop(
            'in_chs'), predictor_cfg.pop('out_chs')
        fc_cfg = predictor_cfg.pop('fc')
        assert num_fcs == len(fc_cfg) == len(in_chs) == len(out_chs)
        predictor = []
        for i in range(num_fcs):
            _cfg = fc_cfg[i]
            _cfg['linear']['in_features'] = in_chs[i]
            _cfg['linear']['out_features'] = out_chs[i]
            predictor.append(LinearModule(**_cfg))
            if i == num_fcs - 1:
                self.vocab_size = out_chs[i]
        assert self.vocab_size == self.num_bin + 1
        self.end = self.vocab_size - 1
        self.predictor = nn.Sequential(*predictor)

        if multi_task:
            # bbox_token, x1, y1, x2, y2, mask_token, x1, y1, ..., xN, yN
            self.task_embedding = nn.Embedding(2, self.d_model)

        self.transformer._init_layers(in_ch,
                                      self.vocab_size,
                                      x_positional_encoding,
                                      seq_positional_encoding)

    def quantize(self, seq, img_metas):
        if self.mapping == "relative":
            num_pts = seq.size(1) // 2
            norm_factor = [img_meta['pad_shape'][:2][::-1]
                           for img_meta in img_metas]
            norm_factor = seq.new_tensor(norm_factor)
            norm_factor = torch.cat(
                [norm_factor for _ in range(num_pts)], dim=1)
            return (seq / norm_factor * self.num_bin).long()
        elif self.mapping == "absolute":
            return (seq / 640. * self.num_bin).long()

    def dequantize(self, seq, scale_factor):
        if self.mapping == "relative":
            return seq * scale_factor / self.num_bin
        elif self.mapping == "absolute":
            return seq * 640. / self.num_bin

    def forward(self, x_mm, x_mask, gt_bbox=None, gt_mask_vertices=None):
        with_bbox = gt_bbox is not None
        with_mask = gt_mask_vertices is not None

        x_mask, x_pos_embeds = self.transformer.x_mask_pos_enc(x_mm, x_mask)

        memory = self.transformer.forward_encoder(x_mm, x_mask, x_pos_embeds)

        seq_in_embeds, targets = self.sequentialize(
            img_metas,
            gt_bbox=gt_bbox,
            gt_mask_vertices=gt_mask_vertices)
        logits = self.transformer.forward_decoder(seq_in_embeds, memory, x_pos_embeds, x_mask)
        logits = self.predictor(logits)

        # training statistics
        with torch.no_grad():
            if with_mask and with_bbox:
                logits_bbox = logits[:, :4, :-1]
                scores_bbox = F.softmax(logits_bbox, dim=-1)
                _, seq_out_bbox = scores_bbox.max(
                    dim=-1, keepdim=False)
                logits_mask = logits[:, 5:, :]
                scores_mask = F.softmax(logits_mask, dim=-1)
                _, seq_out_mask = scores_mask.max(
                    dim=-1, keepdim=False)
                return dict(seq_out_bbox=seq_out_bbox.detach(),
                            seq_out_mask=seq_out_mask.detach())
            else:
                if with_bbox:
                    logits = logits[:, :-1, :-1]
                scores = F.softmax(logits, dim=-1)
                _, seq_out = scores.max(dim=-1, keepdim=False)

                if with_bbox:
                    return dict(seq_out_bbox=seq_out.detach())
                elif with_mask:
                    return dict(seq_out_mask=seq_out.detach())

    def forward_test(self, x_mm, img_metas, with_bbox=False, with_mask=False):
        x_mask, x_pos_embeds = self.transformer.x_mask_pos_enc(x_mm, img_metas)
        memory = self.transformer.forward_encoder(x_mm, x_mask, x_pos_embeds)
        return self.generate_sequence(memory, x_mask, x_pos_embeds,
                                      with_bbox=with_bbox,
                                      with_mask=with_mask)

    def generate(self, seq_in_embeds, memory, x_pos_embeds, x_mask, decode_steps, with_mask):
        seq_out = []
        for step in range(decode_steps):
            out = self.transformer.forward_decoder(
                seq_in_embeds, memory, x_pos_embeds, x_mask)
            logits = out[:, -1, :]
            logits = self.predictor(logits)
            if self.multi_task:
                if step < 4:
                    logits = logits[:, :-1]
            else:
                if not with_mask:
                    logits = logits[:, :-1]
            probs = f.softmax(logits, dim=-1)
            if self.top_p > 0.:
                sorted_score, sorted_idx = torch.sort(
                    probs, descending=True)
                cum_score = sorted_score.cumsum(dim=-1)
                sorted_idx_to_remove = cum_score > self.top_p
                sorted_idx_to_remove[...,
                1:] = sorted_idx_to_remove[..., :-1].clone()
                sorted_idx_to_remove[..., 0] = 0
                idx_to_remove = sorted_idx_to_remove.scatter(
                    1, sorted_idx, sorted_idx_to_remove)
                probs = probs.masked_fill(idx_to_remove, 0.)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = probs.max(dim=-1, keepdim=True)

            seq_in_embeds = torch.cat(
                [seq_in_embeds, self.transformer.query_embedding(next_token)], dim=1)

            seq_out.append(next_token)

        seq_out = torch.cat(seq_out, dim=-1)

        return seq_out

    def generate_sequence(self, memory, x_mask, x_pos_embeds, with_bbox=False, with_mask=False):
        """Args:
            memory (tensor): encoder's output, [batch_size, h*w, d_model].

            x_mask (tensor): [batch_size, h*w], dtype is torch.bool, True means
                ignored position.

            x_pos_embeds (tensor): [batch_size, h*w, d_model].
        """
        batch_size = memory.size(0)
        if with_bbox and with_mask:
            task_bbox = self.task_embedding.weight[0].unsqueeze(
                0).unsqueeze(0).expand(batch_size, -1, -1)
            seq_out_bbox = self.generate(
                task_bbox, memory, x_pos_embeds, x_mask, 4, False)
            task_mask = self.task_embedding.weight[1].unsqueeze(
                0).unsqueeze(0).expand(batch_size, -1, -1)
            seq_in_embeds_box = self.transformer.query_embedding(
                seq_out_bbox)
            seq_in_embeds_mask = torch.cat(
                [task_bbox, seq_in_embeds_box, task_mask], dim=1)
            seq_out_mask = self.generate(
                seq_in_embeds_mask, memory, x_pos_embeds, x_mask, 2 * self.num_ray + 1, True)
            return dict(seq_out_bbox=seq_out_bbox,
                        seq_out_mask=seq_out_mask)
        else:
            seq_in_embeds = memory.new_zeros((batch_size, 1, self.d_model))
            if with_mask:
                decode_steps = self.num_ray * 2 + 1
            elif with_bbox:
                decode_steps = 4
            seq_out = self.generate(
                seq_in_embeds, memory, x_pos_embeds, x_mask, decode_steps, with_mask)
            if with_bbox:
                return dict(seq_out_bbox=seq_out)
            elif with_mask:
                return dict(seq_out_mask=seq_out)

    def sequentialize(self,
                      img_metas,
                      gt_bbox=None,
                      gt_mask_vertices=None,
                      ):
        """Args:
            gt_bbox (list[tensor]): [4, ].

            gt_mask_vertices (tensor): [batch_size, 2 (in x, y order), num_ray].
        """
        with_bbox = gt_bbox is not None
        with_mask = gt_mask_vertices is not None
        assert with_bbox or with_mask
        batch_size = len(img_metas)

        if with_bbox:
            seq_in_bbox = torch.vstack(gt_bbox)

        if with_mask:
            seq_in_mask = gt_mask_vertices.transpose(1, 2).reshape(batch_size, -1)

        if with_bbox and with_mask:
            assert self.multi_task
            seq_in = torch.cat([seq_in_bbox, seq_in_mask], dim=-1)
        elif with_bbox:
            seq_in = seq_in_bbox
        elif with_mask:
            seq_in = seq_in_mask

        seq_in = self.quantize(seq_in, img_metas)
        if with_mask:
            seq_in[seq_in < 0] = self.end
        seq_in[seq_in != self.end].clamp_(min=0, max=self.num_bin - 1)

        if with_bbox and with_mask:
            # bbox_token, x1, y1, x2, y2, mask_token, x1, y1, ..., xN, yN
            if self.shuffle_fraction > 0.:
                seq_in[:, 4:] = self.shuffle_sequence(seq_in[:, 4:])
            seq_in_bbox, seq_in_mask = torch.split(
                seq_in, [4, seq_in.size(1) - 4], dim=1)
            targets = torch.cat([seq_in_bbox, seq_in_bbox.new_full(
                (batch_size, 1), self.end), seq_in_mask, seq_in_mask.new_full((batch_size, 1), self.end)], dim=-1)
            seq_in_embeds_bbox = self.transformer.query_embedding(
                seq_in_bbox)
            seq_in_embeds_mask = self.transformer.query_embedding(
                seq_in_mask)
            task_bbox = self.task_embedding.weight[0].unsqueeze(
                0).unsqueeze(0).expand(batch_size, -1, -1)
            task_mask = self.task_embedding.weight[1].unsqueeze(
                0).unsqueeze(0).expand(batch_size, -1, -1)
            seq_in_embeds = torch.cat(
                [task_bbox, seq_in_embeds_bbox, task_mask, seq_in_embeds_mask], dim=1)
            return seq_in_embeds, targets
        else:
            if with_mask and self.shuffle_fraction > 0.:
                seq_in = self.shuffle_sequence(seq_in)
            seq_in_embeds = self.transformer.query_embedding(seq_in)
            targets = torch.cat(
                [seq_in, seq_in.new_full((batch_size, 1), self.end)], dim=-1)
            seq_in_embeds = torch.cat(
                [seq_in_embeds.new_zeros((batch_size, 1, self.d_model)), seq_in_embeds], dim=1)
            return seq_in_embeds, targets