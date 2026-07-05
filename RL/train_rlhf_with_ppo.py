import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# ================================
# 模型结构定义
# ================================
class SimpleGPT(nn.Module):
    """使用 Transformer 的语言模型（GPT）"""
    def __init__(self, vocab_size=10000, max_seq_len=1024, hidden_size=768, num_layers=12, num_heads=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)    # Token 嵌入
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)    # 位置编码（可学习）
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)    # Transformer Decoder 层（GPT使用decoder，因果注意力）
        ])
        self.ln = nn.LayerNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        device = input_ids.device

        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)    # 位置编码
        x = self.token_embedding(input_ids) + self.position_embedding(positions)    # 嵌入 + 位置编码
        for layer in self.layers:
            x = layer(x)    # 通过 Transformer 层
        x = self.ln(x)    # 最终层归一化
        logits = self.output_head(x)    # 输出 logits
        return logits
    
    def get_hidden(self, input_ids):
        """获取隐藏状态"""
        seq_len = input_ids.shape[1]
        device = input_ids.device

        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        return x  # [batch, seq, hidden_size]
    
    def log_prob(self, prompts, responses):
        """计算 log probability"""
        combined = torch.cat([prompts, responses], dim=1)
        logits = self.forward(combined)
        log_probs = F.log_softmax(logits, dim=-1)
        response_log_probs = log_probs[:, prompts.shape[1]-1:-1, :]   # 只返回 response 部分 (最后一个 prompt token ~ 倒数第二个 token) 的 log prob
        return response_log_probs.gather(2, responses.unsqueeze(-1)).squeeze(-1)    # 提取模型对 response 部分每个 token 的预测对数概率
    
    def generate(self, prompts, max_length=20, temperature=1.0):
        """自回归生成"""
        was_training = self.training
        self.eval()
        generated = prompts.clone()
        with torch.no_grad():
            for _ in range(max_length):    # 循环生成 max_length 个 token，每次生成一个 token
                logits = self.forward(generated)  # [batch, current_seq_len] -> [batch, current_seq_len, vocab_size] 每个位置对下一个token的预测分数
                next_token_logits = logits[:, -1, :] / temperature   # 取最后一个位置的预测，除以 temperature 控制随机性 (<1 放大差异，更确定，>1 缩小差异，更随机)
                probs = F.softmax(next_token_logits, dim=-1)    # 将 logits 转换为概率分布，[batch, vocab_size]
                next_token = torch.multinomial(probs, num_samples=1)   # 根据概率分布采样，返回 [batch, 1] 每个样本采样的 token ID
                generated = torch.cat([generated, next_token], dim=1)   # 将新生成的 token 拼接到序列末尾

        if was_training:
            self.train()

        return generated[:, prompts.shape[1]:]   # 只返回生成的部分（去除 prompt）


class TransformerBlock(nn.Module):
    """单个 Transformer 块"""
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)   # 输入形状 [batch, seq, dim]
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", None)    # 因果掩码（不需要训练，因为是固定的规则，即模型看不到未来的token）
        
    def forward(self, x):
        # 创建因果掩码
        if self.causal_mask is None or self.causal_mask.shape[-1] != x.shape[1]:
            seq_len = x.shape[1]
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            self.causal_mask = mask
        # MHA
        attn_output, attn_weight = self.attention(x, x, x, attn_mask=self.causal_mask)
        x = self.norm1(x + self.dropout(attn_output))
        # FFN
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class RewardModel(nn.Module):
    """奖励模型：使用 GPT 作为 backbone
       默认不冻结 backbone，允许其参数在训练过程中被更新，即总共可训练：7B (backbone) + 769 (hidden_size + 1) ≈ 7B 参数"""
    def __init__(self, base_model, freeze_backbone=False):
        super().__init__()
        # 独立一份 backbone，避免和外部共享参数
        self.backbone = copy.deepcopy(base_model)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.reward_head = nn.Linear(base_model.hidden_size, 1)

    def forward(self, prompts, responses):
        combined = torch.cat([prompts, responses], dim=1)
        hidden_states = self.backbone.get_hidden(combined)
        last_hidden = hidden_states[:, -1, :]
        return self.reward_head(last_hidden).squeeze(-1)   # [batch_size, 1]


class ValueModel(nn.Module):
    """PPO 价值模型：GPT backbone + 标量 value head"""
    def __init__(self, base_model):
        super().__init__()
        self.backbone = copy.deepcopy(base_model)
        self.value_head = nn.Linear(base_model.hidden_size, 1)

    def forward(self, input_ids):    # 输入是 prompts + responses
        hidden_states = self.backbone.get_hidden(input_ids)
        return self.value_head(hidden_states).squeeze(-1)    # 返回每个 token 位置的标量价值 [batch, seq_len]


# ================================
# 定义配置类
# ================================
class Config:
    """RLHF 训练配置"""
    # 模型参数
    vocab_size = 10000
    hidden_size = 768
    num_layers = 12
    num_heads = 12
    max_seq_len = 1024
    
    # PPO 参数
    ppo_beta = 0.1          # KL 惩罚系数
    ppo_epsilon = 0.2       # PPO clip 范围
    ppo_epochs = 4          # PPO 更新轮数
    ppo_gamma = 0.99        # 折扣因子
    ppo_lam = 0.95          # GAE lambda
    
    # 训练参数
    learning_rate = 1e-5
    batch_size = 4
    num_ppo_iterations = 10
    
    # 序列长度
    response_length = 20


# ================================
# 辅助函数
# ================================
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """计算广义优势估计"""
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    
    for b in range(batch_size):
        gae = 0
        rewards_b = rewards[b].detach().cpu().numpy()
        values_b = values[b].detach().cpu().numpy()

        for t in reversed(range(seq_len)):
            if t + 1 < seq_len:
                delta = rewards_b[t] + gamma * values_b[t+1] - values_b[t]   # TD 误差
            else:
                delta = rewards_b[t] - values_b[t]
            gae = delta + gamma * lam * gae
            advantages[b, t] = gae

    return advantages


def compute_discounted_returns(rewards, gamma=0.99):
    """计算折扣累积奖励"""
    batch_size, seq_len = rewards.shape
    returns = torch.zeros_like(rewards)
    
    for b in range(batch_size):
        R = 0
        rewards_b = rewards[b].detach().cpu().numpy()
        
        for t in reversed(range(seq_len)):
            R = rewards_b[t] + gamma * R
            returns[b, t] = R
    
    return returns


# ================================
# 训练函数
# ================================
def train_policy_with_ppo(policy_model, ref_policy, reward_model, value_model, prompts, responses, config):
    """
    PPO 训练策略模型 (policy_model 和 value_model 一起训练)
    """
    policy_model.train()

    device = next(policy_model.parameters()).device   # 取模型所有参数中的第一个参数，查看该参数所在的设备
    prompts = prompts.to(device)
    responses = responses.to(device)

    beta = config.ppo_beta
    epsilon = config.ppo_epsilon
    ppo_epochs = config.ppo_epochs
    
    # 先计算 ref_log_probs 和 旧策略 log_probs（都不需要梯度）
    with torch.no_grad():
        ref_log_probs = ref_policy.log_prob(prompts, responses)
        old_log_probs = policy_model.log_prob(prompts, responses)

    # 计算奖励和 KL 惩罚
    with torch.no_grad():
        rewards = reward_model(prompts, responses)
        kl_divs = old_log_probs - ref_log_probs
        token_rewards = -beta * kl_divs  # [batch, response_len]
        token_rewards[:, -1] += rewards

        # 获取价值
        combined = torch.cat([prompts, responses], dim=1)  # [batch, prompt_len + response_len]
        response_len = responses.shape[1]
        all_values = value_model(combined)              # [batch, prompt_len + response_len]
        values_scalar = all_values[:, -response_len:]   # [batch, response_len]

        # 计算优势
        advantages = compute_gae(token_rewards, values_scalar, gamma=config.ppo_gamma, lam=config.ppo_lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)    # advantage 归一化（稳定训练）

        # 计算 returns (使用 GAE 估计的加权折扣回报)
        returns = advantages + values_scalar

        # 另一种: 使用蒙特卡洛折扣回报 (简单的折扣回报)
        # returns = compute_discounted_returns(token_rewards, gamma=config.ppo_gamma)
        # advantages = returns - values_scalar

    # PPO 更新
    for _ in range(ppo_epochs):
        new_log_probs = policy_model.log_prob(prompts, responses)
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        combined = torch.cat([prompts, responses], dim=1)
        all_new_values = value_model(combined)     # [batch, prompt_len + response_len]
        new_values_scalar = all_new_values[:, -response_len:]
        value_loss = F.mse_loss(new_values_scalar, returns)
        
        total_loss = policy_loss + 0.5 * value_loss
        
        # 反向传播
        policy_model.optimizer.zero_grad()
        value_model.optimizer.zero_grad()

        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), 1.0)

        policy_model.optimizer.step()
        value_model.optimizer.step()
    
    return policy_model


def train_reward_model(reward_model, comparison_data):
    """训练奖励模型"""
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-5)
    device = next(reward_model.parameters()).device
    
    for batch in comparison_data:
        prompts = batch["prompt"].to(device)
        chosen = batch["chosen"].to(device)
        rejected = batch["rejected"].to(device)
        
        reward_chosen = reward_model(prompts, chosen)
        reward_rejected = reward_model(prompts, rejected)
        
        loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return reward_model


def train_reference_policy(base_model, sft_data):
    """SFT 训练参考策略"""
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-5)
    device = next(base_model.parameters()).device
    
    for batch in sft_data:
        prompts = batch["prompt"].to(device)
        responses = batch["response"].to(device)
        combined = torch.cat([prompts, responses], dim=1)
        logits = base_model(combined)
        
        # 只计算 response 部分的损失
        response_logits = logits[:, prompts.shape[1]-1:-1, :]
        response_targets = responses
        loss = F.cross_entropy(    # 期望输入为 [N, C]，其中 N=样本数、C=类别数
            response_logits.reshape(-1, response_logits.size(-1)),
            response_targets.reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return base_model


# ================================
# 主训练流程
# ================================
def main():
    """主训练函数"""
    config = Config()  # 创建配置实例

    print("=" * 50)
    print("开始 RLHF 训练流程")
    print("=" * 50)
    
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 2. 初始化模型
    print("\n[步骤1] 初始化模型 base_model ...")
    base_model = SimpleGPT(
        vocab_size=config.vocab_size, 
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads
    )
    base_model.to(device)
    base_model.optimizer = torch.optim.AdamW(base_model.parameters(), lr=config.learning_rate)
    
    # 3. 准备训练数据
    print("\n[步骤2] 准备训练数据 sft_data & compare_pair_data ...")
    sft_data = [
        {"prompt": torch.randint(0, 10000, (1, 10)).to(device), 
         "response": torch.randint(0, 10000, (1, 20)).to(device)},
        {"prompt": torch.randint(0, 10000, (1, 10)).to(device), 
         "response": torch.randint(0, 10000, (1, 20)).to(device)},
    ]

    comparison_data = [
        {
            "prompt": torch.randint(0, 10000, (1, 10)).to(device),
            "chosen": torch.randint(0, 10000, (1, 20)).to(device),
            "rejected": torch.randint(0, 10000, (1, 20)).to(device)
        },
    ]
    
    # 4. 第一阶段：训练参考策略（SFT）
    print("\n[步骤3] 第一阶段：监督微调训练 参考策略 Reference Policy ...")
    sft_model = train_reference_policy(base_model, sft_data)
    print("✓ SFT 训练完成")

    # ref_policy 必须独立，并冻住（PPO 用作 KL 参考）
    ref_policy = copy.deepcopy(sft_model).to(device)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    # 5. 第二阶段：训练奖励模型（独立 backbone）
    print("\n[步骤4] 第二阶段：训练奖励模型...")
    reward_model = RewardModel(sft_model).to(device)
    reward_model = train_reward_model(reward_model, comparison_data)
    print("✓ 奖励模型训练完成")

    # 6. 第三阶段：初始化策略模型和价值模型
    print("\n[步骤5] 第三阶段：初始化 PPO 训练组件...")
    
    # Policy model：从 SFT 权重初始化，参数独立
    policy_model = copy.deepcopy(sft_model).to(device)
    policy_model.optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate)

    # Value model：从 SFT 初始化 backbone，加独立 value head
    value_model = ValueModel(sft_model).to(device)
    value_model.optimizer = torch.optim.AdamW(value_model.parameters(), lr=config.learning_rate)
    print("✓ 策略模型和价值模型初始化完成")
    
    # 7. 准备 PPO 训练数据
    print("\n[步骤6] 准备 PPO 训练数据...")
    batch_size = 4
    seq_len_prompt = 10
    prompts = torch.randint(0, 10000, (batch_size, seq_len_prompt)).to(device)
    
    # 8. 第四阶段：PPO 训练
    print("\n[步骤7] 第四阶段：开始 PPO 训练...")
    for iteration in range(config.num_ppo_iterations):
        print(f"\n--- PPO 迭代 {iteration + 1}/{config.num_ppo_iterations} ---")
        
        responses = policy_model.generate(prompts, max_length=config.response_length)
        policy_model = train_policy_with_ppo(
            policy_model=policy_model,
            ref_policy=ref_policy,
            reward_model=reward_model,
            value_model=value_model,
            prompts=prompts,
            responses=responses,
            config=config  # 传入配置
        )
        
        # 定期保存检查点
        if (iteration + 1) % 5 == 0:
            print(f"\n保存检查点 at iteration {iteration + 1}")
            torch.save({
                'policy_model': policy_model.state_dict(),
                'value_model': value_model.state_dict(),
                'iteration': iteration
            }, f'checkpoint_iter_{iteration + 1}.pt')
    
    # 9. 保存最终模型
    print("\n[步骤8] 保存最终模型...")
    torch.save({
        'policy_model': policy_model.state_dict(),
        'ref_policy': ref_policy.state_dict(),
        'reward_model': reward_model.state_dict(),
        'value_model': value_model.state_dict(),
    }, 'final_rlhf_model.pt')
    
    print("\n" + "=" * 50)
    print("RLHF 训练完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()