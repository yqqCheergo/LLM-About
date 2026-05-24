import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt((x ** 2).mean(-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm  # 无beta参数