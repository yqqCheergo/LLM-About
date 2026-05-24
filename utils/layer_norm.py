import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # 可学习的缩放参数，调整方差，默认为1
        self.beta = nn.Parameter(torch.zeros(d_model))  # 可学习的偏移参数，调整均值，默认为0
        self.eps = eps

    def forward(self, x):
        # x: [batch_size, seq_len, d_model] 或 [batch_size, d_model]

        # 1. 在最后一个维度上计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)  # 保持维度，方便广播
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 有偏估计

        # 2. 归一化
        x_hat = (x - mean) / torch.sqrt(var + self.eps)    # 减去均值除以标准差

        # 3. 缩放和偏移（广播自动处理）
        return self.gamma * x_hat + self.beta    # 再施以线性映射