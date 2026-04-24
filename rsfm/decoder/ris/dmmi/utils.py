from torch import nn


class Transformer_Fusion(nn.Module):
    def __init__(self, dim=768, nhead=8, num_layers=1):
        super(Transformer_Fusion, self).__init__()

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=nhead)
        self.transformer_model = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    def forward(self, vis, lan_full):
        WW, HH = vis.shape[2], vis.shape[3]
        vis = vis.view(vis.shape[0], vis.shape[1], -1)
        vis = vis.permute(2, 0, 1)
        lan = lan_full.permute(2, 0, 1)
        vis = self.transformer_model(vis, lan)
        vis = vis.permute(1, 2, 0)
        vis = vis.view(vis.shape[0], vis.shape[1], WW, HH)

        return vis