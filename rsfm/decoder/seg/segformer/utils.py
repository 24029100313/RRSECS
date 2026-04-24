import torch.nn as nn
from torch import Tensor



class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x:Tensor)->Tensor:
        # equivalent to conv1x1?
        x = x.flatten(2).transpose(1, 2) # (B, C, H, W) -> (B, H*W, C)
        x = self.proj(x)
        return x