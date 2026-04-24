import torch
from torch import nn
import math



class Attention(nn.Module):
    def __init__(self, cross_vis_hidden_size, cross_text_hidden_size, cross_num_attention_heads,
                 cross_attention_probs_dropout_prob, query_vis=True):
        super().__init__()
        if query_vis:
            self.cross_hidden_size = cross_vis_hidden_size
            self.cm_hidden_size = cross_text_hidden_size
        else:
            self.cross_hidden_size = cross_text_hidden_size
            self.cm_hidden_size = cross_vis_hidden_size

        if self.cross_hidden_size % cross_num_attention_heads != 0 and \
                self.cm_hidden_size % cross_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.cross_hidden_size, cross_num_attention_heads))

        self.cross_num_attention_heads = cross_num_attention_heads
        self.cross_attention_head_size = int(self.cross_hidden_size / cross_num_attention_heads)
        self.query_vis = query_vis
        self.cross_all_head_size = self.cross_num_attention_heads * self.cross_attention_head_size

        self.query = nn.Linear(self.cross_hidden_size, self.cross_all_head_size)
        self.key = nn.Linear(self.cm_hidden_size, self.cross_all_head_size)
        self.value = nn.Linear(self.cm_hidden_size, self.cross_all_head_size)

        self.dropout = nn.Dropout(cross_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.cross_num_attention_heads, self.cross_attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        if not self.query_vis:
            mixed_key_layer = mixed_key_layer.permute(1, 0, 2)
            mixed_value_layer = mixed_value_layer.permute(1, 0, 2)
        else:
            mixed_query_layer = mixed_query_layer.permute(1, 0, 2)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.cross_attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(1)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.cross_all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.query_vis:
            context_layer = context_layer.permute(1, 0, 2)
        return context_layer


class AttOutput(nn.Module):
    def __init__(self, cross_vis_hidden_size, cross_text_hidden_size, cross_hidden_dropout_prob, query_vis=True):
        super(AttOutput, self).__init__()
        if query_vis:
            self.dense = nn.Linear(cross_vis_hidden_size, cross_vis_hidden_size)
            self.LayerNorm = nn.LayerNorm(cross_vis_hidden_size, eps=1e-12)
        else:
            self.dense = nn.Linear(cross_text_hidden_size, cross_text_hidden_size)
            self.LayerNorm = nn.LayerNorm(cross_text_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(cross_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class cross_module(nn.Module):
    def __init__(self, 
                 num_features=1,
                 cross_num_att_heads=1,
                 cross_vis_dim=256,
                 cross_text_dim=768,
                 cross_hid_dropout=0.1,
                 cross_att_dropout=0.1):
        super().__init__()
        self.lang_att = Attention(cross_vis_dim, cross_text_dim, cross_num_att_heads, cross_att_dropout, False)
        self.lang_output = AttOutput(cross_vis_dim, cross_text_dim, cross_hid_dropout, False)
        self.number = num_features  # visual feature number
        self.vis_att = Attention(cross_vis_dim, cross_text_dim, cross_num_att_heads, cross_att_dropout)
        self.vis_output = AttOutput(cross_vis_dim, cross_text_dim, cross_hid_dropout)
        self.vis_liner = nn.Linear(self.number * cross_vis_dim, cross_vis_dim)
        self.lang_liner = nn.Linear(self.number * cross_text_dim, cross_text_dim)

    def act(self, input_tensor, ctx_tensor, query_vis=True, ctx_att_mask=None):
        if query_vis:
            output = self.vis_att(input_tensor, ctx_tensor, ctx_att_mask)
            attention_output = self.vis_output(output, input_tensor)
        else:
            output = self.lang_att(input_tensor, ctx_tensor, ctx_att_mask)
            attention_output = self.lang_output(output, input_tensor)
        return attention_output

    def forward(self, vis_input, lang_input, vis_attention_mask, lang_attention_mask):
        # lang_output, vis_output = [], []
        #
        # for i in range(self.number):
        #     lang_att_output = self.act(lang_input, vis_input[i], False, ctx_att_mask=vis_attention_mask)
        #     vis_att_output = self.act(vis_input[i], lang_input, ctx_att_mask=lang_attention_mask)
        #     lang_output.append(lang_att_output)
        #     vis_output.append(vis_att_output)
        #
        # lang_output = self.lang_liner(torch.cat(lang_output, 2))
        # vis_output = self.vis_liner(torch.cat(vis_output, 2))
        #
        # return lang_output, vis_output


        lang_output = self.act(lang_input, vis_input, False, ctx_att_mask=vis_attention_mask)
        vis_output = self.act(vis_input, lang_input, ctx_att_mask=lang_attention_mask)

        lang_output = self.lang_liner(lang_output)
        vis_output = self.vis_liner(vis_output)

        return lang_output, vis_output
