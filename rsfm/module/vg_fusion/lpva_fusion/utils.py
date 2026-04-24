import torch.nn as nn
from torch.nn import functional as F



class PA(nn.Module):
    def __init__(self, v_channels=96, l_channels=768, size=16):
        super(PA, self).__init__()
        self.size = size

        self.text_proj = nn.Linear(l_channels, 256)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.gamma_mlp = nn.Sequential(nn.Linear(256, 256),
                                       nn.Tanh(),
                                       nn.Linear(256, v_channels))

        self.beta_mlp = nn.Sequential(nn.Linear(256, 256),
                                       nn.Tanh(),
                                       nn.Linear(256, v_channels))

        if judge(size):
            base_k = 1
            k = s = size // 16
        else:
            base_k = 5
            k = s = size // 20

        self.rafa_mlp = nn.Sequential(nn.ConvTranspose2d(1, 1, kernel_size=base_k, stride=1, padding=0),
                                      nn.Tanh(),
                                      nn.ConvTranspose2d(1, 1, kernel_size=k if k != 0 else 1, stride=s if s != 0 else 1))

    def forward(self, x, l):
        b = x.shape[0]
        x = x.permute(0, 2, 1).reshape(b, -1, self.size, self.size)

        l = self.text_proj(l)

        ef = self.avgpool(l.permute(0, 2, 1)).squeeze(-1) # B, 256

        gamma = self.gamma_mlp(ef).view(b, -1, 1, 1) # B, 96, 1, 1
        beta = self.beta_mlp(ef).view(b, -1, 1, 1) # B, 96, 1, 1

        out1 = gamma * x + beta

        weight = ef.view(b, 16, 16).unsqueeze(1)
        weight = self.rafa_mlp(weight)

        out2 = x * weight
        out = F.relu(out1 + out2)

        out = x + out

        return out.flatten(2).permute(0, 2, 1)


def judge(size):
    if size in [128, 64, 32, 16]:
        return True
    elif size in [160, 80, 40, 20]:
        return False
    else:
        return NotImplementedError