from .modeling_super_kd import SuperTinyBertForPreTraining, BertConfig
from torch import nn


__all__ = ['AutoTinyBertModel']


class AutoTinyBertModel(nn.Module):
    def __init__(self, pretrained_path):
        super(AutoTinyBertModel, self).__init__()

        config = BertConfig.from_pretrained(pretrained_path)
        self.model = SuperTinyBertForPreTraining.from_pretrained(pretrained_path, config)

        self.submodel_config = dict()
        self.submodel_config['sample_layer_num'] = config.num_hidden_layers
        self.submodel_config['sample_hidden_size'] = config.hidden_size
        self.submodel_config['sample_intermediate_sizes'] = config.num_hidden_layers * [config.intermediate_size]
        self.submodel_config['sample_num_attention_heads'] = config.num_hidden_layers * [config.num_attention_heads]
        self.submodel_config['sample_qkv_sizes'] = config.num_hidden_layers * [config.qkv_size]

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, kd=True):
        last_rep, last_att = self.model(input_ids, self.submodel_config, token_type_ids, attention_mask, kd=kd)

        return last_rep, last_att
