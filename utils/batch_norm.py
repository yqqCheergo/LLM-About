import torch
import numpy as np
import torch.nn as nn


class BN:
    def __init__(self, num_features, momentum=0.1, eps=1e-5, track_running_stats=True):
        """
        Args:
            num_features: 特征通道数
            momentum: 当前batch统计量的权重（与PyTorch一致）
            eps: 防止除零
            track_running_stats: 是否追踪全局统计量
        """
        # 可学习参数
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # 移动平均统计量（推理使用）
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.momentum = momentum
        self.eps = eps
        self.track_running_stats = track_running_stats
        self.training = True

    def forward(self, x):
        """
        x: [batch, channels, height, width]
        """
        if self.training:
            # 1. 计算当前batch的均值和方差（训练用）
            batch_mean = x.mean(axis=(0, 2, 3))  # [C]
            # 有偏估计（分母=m），用于归一化
            batch_var_biased = x.var(axis=(0, 2, 3), ddof=0)  # ddof=0: 分母为m
            # 无偏估计（分母=m-1），用于更新running_var
            batch_var_unbiased = x.var(axis=(0, 2, 3), ddof=1)  # ddof=1: 分母为m-1

            # [C] -> [1, C, 1, 1] 使其能与 [N, C, H, W] 广播
            batch_mean_reshaped = batch_mean.reshape(1, -1, 1, 1)
            batch_var_biased_reshaped = batch_var_biased.reshape(1, -1, 1, 1)

            # 2. 归一化当前batch（使用有偏估计）
            x_hat = (x - batch_mean_reshaped) / np.sqrt(batch_var_biased_reshaped + self.eps)

            # 3. 更新移动平均（使用无偏估计）
            if self.track_running_stats:
                # PyTorch公式：new = (1 - momentum) * old + momentum * current
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var_unbiased

        else:
            # 推理时使用移动平均
            running_mean_reshaped = self.running_mean.reshape(1, -1, 1, 1)
            running_var_reshaped = self.running_var.reshape(1, -1, 1, 1)
            x_hat = (x - running_mean_reshaped) / np.sqrt(running_var_reshaped + self.eps)

        # 4. 缩放和平移（可学习参数）
        gamma_reshaped = self.gamma.reshape(1, -1, 1, 1)
        beta_reshaped = self.beta.reshape(1, -1, 1, 1)
        return gamma_reshaped * x_hat + beta_reshaped

    def eval(self):
        """切换到推理模式"""
        self.training = False

    def train(self):
        """切换到训练模式"""
        self.training = True


# 验证与PyTorch的一致性
def vs_pytorch():

    torch.manual_seed(42)
    np.random.seed(42)

    # 测试数据 (2个样本, 3个通道, 4x4特征图)
    x_np = np.random.randn(2, 3, 4, 4).astype(np.float32)
    x_torch = torch.from_numpy(x_np)

    ##### PyTorch BN层 #####
    bn_torch = nn.BatchNorm2d(3, momentum=0.1, eps=1e-5, affine=True)  # affine 表示是否使用可学习的缩放和平移参数
    bn_torch.train()

    # 手动设置相同的初始参数
    bn_torch.weight.data = torch.ones(3)
    bn_torch.bias.data = torch.zeros(3)
    bn_torch.running_mean.data = torch.zeros(3)
    bn_torch.running_var.data = torch.ones(3)

    # PyTorch前向传播
    out_torch = bn_torch(x_torch)

    ##### 自定义BN层 #####
    bn_custom = BN(3, momentum=0.1, eps=1e-5)

    # 自定义BN前向传播
    out_custom = bn_custom.forward(x_np)

    # 比较输出
    print("输出差异:", np.max(np.abs(out_custom - out_torch.detach().numpy())))
    print("running_mean 差异:", np.max(np.abs(bn_custom.running_mean - bn_torch.running_mean.numpy())))
    print("running_var 差异:", np.max(np.abs(bn_custom.running_var - bn_torch.running_var.numpy())))

    # 推理模式对比
    bn_torch.eval()
    bn_custom.eval()

    out_torch_eval = bn_torch(x_torch)
    out_custom_eval = bn_custom.forward(x_np)

    print("推理输出差异:", np.max(np.abs(out_custom_eval - out_torch_eval.detach().numpy())))

    # 再次训练模式验证
    bn_torch.train()
    bn_custom.train()
    out_torch_train2 = bn_torch(x_torch)
    out_custom_train2 = bn_custom.forward(x_np)
    print("第二次训练输出差异:", np.max(np.abs(out_custom_train2 - out_torch_train2.detach().numpy())))


if __name__ == "__main__":
    vs_pytorch()