import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.utils.data import DataLoader

# ======================== 共享组件 ========================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        normed_x = self.norm1(x)
        attn_output = self.self_attn(normed_x, normed_x, normed_x, mask)
        x = x + self.dropout(attn_output)
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout(ff_output)
        return x


# ======================== 模型定义 ========================

class GPTModel(nn.Module):
    """基础GPT模型，用于SFT阶段的策略模型"""
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_length, d_ff, dropout=0.1):
        super(GPTModel, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_length, d_model)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        scale = 1.0 / math.sqrt(self.num_layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for layer in self.decoder_layers:
            layer.self_attn.W_o.weight.data *= scale
            layer.feed_forward.fc2.weight.data *= scale

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(
            input_ids.size(0), 1)
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(positions)
        embeddings = self.dropout(token_embeds + position_embeds)

        mask = torch.tril(torch.ones(seq_length, seq_length, device=input_ids.device)).unsqueeze(0).unsqueeze(0)

        x = embeddings
        for layer in self.decoder_layers:
            x = layer(x, mask)

        logits = self.fc(x)
        return logits


class RewardModel(nn.Module):
    """奖励模型：基于GPT的transformer层，输出标量奖励分数"""
    def __init__(self, base_model):
        super(RewardModel, self).__init__()
        # 复用base_model的transformer部分（不含最后的fc层）
        self.token_embeddings = base_model.token_embeddings
        self.position_embeddings = base_model.position_embeddings
        self.decoder_layers = base_model.decoder_layers
        self.dropout = base_model.dropout
        # 奖励头：将d_model映射到1维标量（base_model中最后的fc层将d_model映射到vocab_size）
        self.reward_head = nn.Linear(base_model.fc.in_features, 1)

    def forward(self, input_ids):
        # 以下均不变
        seq_length = input_ids.size(1)
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(
            input_ids.size(0), 1)
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(positions)
        embeddings = self.dropout(token_embeds + position_embeds)

        mask = torch.tril(torch.ones(seq_length, seq_length, device=input_ids.device)).unsqueeze(0).unsqueeze(0)

        x = embeddings
        for layer in self.decoder_layers:
            x = layer(x, mask)
        # 以上均不变

        # 取最后一个token的输出作为整个序列的表示，映射到标量奖励
        reward = self.reward_head(x[:, -1, :])  # [batch_size, 1]
        return reward.squeeze(-1)  # [batch_size]


class ValueModel(nn.Module):
    """价值模型：输出每个token位置的状态价值，用于PPO的baseline估计"""
    def __init__(self, base_model):
        super(ValueModel, self).__init__()
        self.token_embeddings = base_model.token_embeddings
        self.position_embeddings = base_model.position_embeddings
        self.decoder_layers = base_model.decoder_layers
        self.dropout = base_model.dropout
        # 价值头：每个token位置输出一个标量
        self.value_head = nn.Linear(base_model.fc.in_features, 1)

    def forward(self, input_ids):
        # 以下均不变
        seq_length = input_ids.size(1)
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(
            input_ids.size(0), 1)
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(positions)
        embeddings = self.dropout(token_embeds + position_embeds)

        mask = torch.tril(torch.ones(seq_length, seq_length, device=input_ids.device)).unsqueeze(0).unsqueeze(0)

        x = embeddings
        for layer in self.decoder_layers:
            x = layer(x, mask)
        # 以上均不变

        values = self.value_head(x).squeeze(-1)  # [batch_size, seq_length]
        return values


# ======================== 阶段1: SFT 监督微调 ========================

def train_sft(model, dataloader, criterion, optimizer, device, response_start_idx):
    """
    SFT阶段：在 instruction-response 对上进行监督微调
    只计算response部分的损失（instruction部分不参与损失计算）
    """
    model.train()
    total_loss = 0
    for input_ids in dataloader:
        input_ids = input_ids.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)  # [batch_size, seq_length, vocab_size]

        # 标准next-token prediction：用位置i的logits预测位置i+1的token
        # 最后一个logits丢弃（没有下一个token）
        shift_logits = logits[:, :-1, :].contiguous()   # [batch_size, seq_length[:-1], vocab_size]
        shift_labels = input_ids[:, 1:].contiguous()    # [batch_size, seq_length[1:]]

        # 只计算response部分的损失（从response_start_idx开始）
        # shift_logits[:, response_start_idx-1:, :] 对应预测 response_start_idx 及之后的token
        response_logits = shift_logits[:, response_start_idx - 1:, :].contiguous()
        response_labels = shift_labels[:, response_start_idx - 1:].contiguous()

        # 交叉熵损失
        loss = criterion(
            response_logits.view(-1, response_logits.size(-1)),   # [batch_size*(response_len-1), vocab_size] 把批次和序列维度合并，每个token预测变成一个独立的预测任务
            response_labels.view(-1)   # [batch_size*(response_len-1)] 每个预测位置对应一个标签值
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止深度模型训练出现梯度爆炸
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# ======================== 阶段2: 奖励模型训练 ========================

def train_reward_model(model, dataloader, optimizer, device):
    """
    RM阶段：用Bradley-Terry偏好模型训练奖励模型
    输入：chosen（人类偏好的response）和 rejected（人类不偏好的response）
    损失：-log(sigmoid(r_chosen - r_rejected))
    """
    model.train()
    total_loss = 0
    for chosen_ids, rejected_ids in dataloader:
        chosen_ids = chosen_ids.to(device)
        rejected_ids = rejected_ids.to(device)

        optimizer.zero_grad()

        # 分别计算chosen和rejected的奖励分数
        r_chosen = model(chosen_ids)    # [batch_size]
        r_rejected = model(rejected_ids)  # [batch_size]

        # Bradley-Terry偏好模型损失
        # 人类偏chosen的概率 = sigmoid(r_chosen - r_rejected)
        # 最大化这个概率等价于最小化 -log(sigmoid(r_chosen - r_rejected))
        loss = -torch.log(torch.sigmoid(r_chosen - r_rejected) + 1e-8).mean()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# ======================== 阶段3: PPO 强化学习 ========================

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    计算广义优势估计（GAE）和 TD误差
    GAE_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}   衡量从t时刻开始往后，总的优势有多大
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)   衡量这一步实际得到的和之前预期的之间的差距
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0    # 最后一步没有下一个状态
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = last_advantage = delta + gamma * lam * last_advantage

    returns = advantages + values   # GAE中 return = advantage + value
    return advantages, returns


def ppo_step(policy_model, ref_model, value_model, reward_model,
             dataloader, ppo_optimizer, value_optimizer, device,
             clip_eps=0.2, ent_coef=0.01, gamma=0.99, lam=0.95, ppo_epochs=4):
    """
    PPO阶段：用RM的奖励信号通过强化学习优化策略模型

    流程：
    1. 用策略模型生成response
    2. 用RM打分得到奖励
    3. 用Value Model估计baseline
    4. 计算GAE（广义优势估计）
    5. PPO-Clip目标函数更新策略
    6. 更新Value Model
    """

    # policy_model 和 value_model 更新，ref_model 和 reward_model 参数冻结
    policy_model.train()
    ref_model.eval()
    value_model.train()
    reward_model.eval()

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    step_count = 0

    for input_ids in dataloader:
        input_ids = input_ids.to(device)

        with torch.no_grad():
            # 用策略模型生成logits（用于计算当前策略的概率）
            policy_logits = policy_model(input_ids)   # [batch_size, seq_length, vocab_size]
            # 用参考模型生成logits（用于KL惩罚）
            ref_logits = ref_model(input_ids)

            # 计算log概率
            # 位置 i 的 logits 预测的是位置 i+1 的 token，因此丢掉最后一个位置（没有对应的目标 token），logits 取 [:, :-1, :]
            policy_log_probs = F.log_softmax(policy_logits[:, :-1, :], dim=-1)   # [batch, seq-1, vocab]
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)

            # 取实际token的log概率
            # labels 取 [:, 1:] — 丢掉第一个位置（通常是 BOS，不需要预测）
            labels = input_ids[:, 1:].contiguous()  # [batch, seq-1]

            # 在维度2（vocab_size维度）上，根据labels中的token ID进行索引
            # 从每个位置的整个词汇表分布中，取出对应token的概率
            policy_token_log_probs = policy_log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [batch, seq-1]
            ref_token_log_probs = ref_log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

            # 计算KL散度作为奖励惩罚项（防止策略偏离参考模型太远）
            kl_div = policy_token_log_probs - ref_token_log_probs  # [batch, seq-1]

            # RM奖励：取序列最后一个位置的奖励
            rm_reward = reward_model(input_ids)  # [batch]

            # 构造每步奖励：大部分步骤为-KL惩罚，最后一步加上RM奖励
            rewards = -0.01 * kl_div     # [batch, seq-1] 每步的KL惩罚
            rewards[:, -1] += rm_reward  # 最后一步加上RM奖励

            # 用Value Model估计每个token位置的价值
            values = value_model(input_ids)[:, :-1]  # [batch, seq-1]

        # 计算GAE（对batch中每个样本分别计算）
        advantages_list = []
        returns_list = []
        for i in range(input_ids.size(0)):   # 遍历batch
            adv, ret = compute_gae(rewards[i], values[i], gamma, lam)
            advantages_list.append(adv)
            returns_list.append(ret)

        advantages = torch.stack(advantages_list)  # [batch, seq-1]
        returns = torch.stack(returns_list)  # [batch, seq-1]

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO多轮更新
        for _ in range(ppo_epochs):
            # 重新计算当前策略的log概率
            current_logits = policy_model(input_ids)
            current_log_probs = F.log_softmax(current_logits[:, :-1, :], dim=-1)
            current_token_log_probs = current_log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

            # 重要性采样比率
            ratio = torch.exp(current_token_log_probs - policy_token_log_probs.detach())

            # PPO-Clip目标
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2).mean()    # .mean() 就是 E_t (对时间步 t 取期望)

            # 熵正则化（鼓励探索）
            entropy = -(F.softmax(current_logits[:, :-1, :], dim=-1) * current_log_probs).sum(-1).mean()    # 标准的熵公式 -Σ P(x) * log P(x) 对所有token求和

            # 总损失 = 策略损失 - β * 熵（鼓励熵增 = 概率分布的不确定性越大 = 鼓励探索）
            total_loss = policy_loss - ent_coef * entropy

            ppo_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            ppo_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_entropy += entropy.item()

        # 更新Value Model
        current_values = value_model(input_ids)[:, :-1]
        value_loss = F.mse_loss(current_values, returns.detach())   # true_label 是用 GAE 估计的 returns

        value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=1.0)
        value_optimizer.step()

        total_value_loss += value_loss.item()
        step_count += 1

    return total_policy_loss / (step_count * ppo_epochs), total_value_loss / step_count, total_entropy / (step_count * ppo_epochs)


# ======================== 自回归生成 ========================

def generate_text(model, input_ids, max_length, device):
    """自回归文本生成"""
    model.eval()
    input_ids = input_ids.to(device)
    with torch.no_grad():
        for _ in range(max_length - input_ids.size(1)):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    return input_ids


# ======================== 主函数 ========================

if __name__ == '__main__':

    # ======================== 示例参数配置 ========================
    vocab_size = 10000
    d_model = 128
    num_heads = 4
    num_layers = 2
    max_seq_length = 512
    d_ff = 512

    #################### InstructGPT中的实际参数 ####################
    # vocab_size = 50257
    # d_model = 4096      # 175B参数时的维度
    # num_heads = 96
    # num_layers = 96
    # max_seq_length = 2048
    # d_ff = 16384        # d_model * 4
    #################### InstructGPT中的实际参数 ####################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================== 模拟数据 ========================

    # 模拟 instruction-response 数据（SFT阶段）
    # 格式：[instruction tokens] [SEP] [response tokens]
    batch_size = 8
    instruction_len = 20   # instruction部分长度
    response_len = 30      # response部分长度
    seq_len = instruction_len + response_len  # 总长度

    # 生成dummy的 instruction-response 对
    dummy_sft_data = [
        torch.randint(1, vocab_size, (batch_size, seq_len))  # 从1开始，0留给PAD
        for _ in range(10)
    ]

    # 模拟偏好数据（RM阶段）
    # chosen: 人类偏好的response, rejected: 人类不偏好的response
    dummy_preference_data = [
        (
            torch.randint(1, vocab_size, (batch_size, seq_len)),  # chosen
            torch.randint(1, vocab_size, (batch_size, seq_len))   # rejected
        )
        for _ in range(10)
    ]

    # PPO阶段复用SFT的数据格式
    dummy_ppo_data = [
        torch.randint(1, vocab_size, (batch_size, seq_len))
        for _ in range(10)
    ]

    # ======================== 阶段1: SFT 监督微调 ========================
    print("=" * 60)
    print("阶段1: SFT 监督微调")
    print("=" * 60)

    sft_model = GPTModel(vocab_size, d_model, num_heads, num_layers, max_seq_length, d_ff).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD
    sft_optimizer = torch.optim.Adam(sft_model.parameters(), lr=1e-4)

    sft_dataloader = DataLoader(dummy_sft_data, batch_size=None)
    num_epochs_sft = 5
    for epoch in range(num_epochs_sft):
        loss = train_sft(sft_model, sft_dataloader, criterion, sft_optimizer, device, response_start_idx=instruction_len)
        print(f"  Epoch {epoch + 1}/{num_epochs_sft}, Loss: {loss:.4f}")

    # ======================== 阶段2: 训练奖励模型 ========================
    print()
    print("=" * 60)
    print("阶段2: 训练奖励模型 (RM)")
    print("=" * 60)

    # 用SFT模型初始化奖励模型
    reward_model = RewardModel(copy.deepcopy(sft_model)).to(device)
    rm_optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-4)

    rm_dataloader = DataLoader(dummy_preference_data, batch_size=None)
    num_epochs_rm = 5
    for epoch in range(num_epochs_rm):
        loss = train_reward_model(reward_model, rm_dataloader, rm_optimizer, device)
        print(f"  Epoch {epoch + 1}/{num_epochs_rm}, Loss: {loss:.4f}")

    # ======================== 阶段3: PPO 强化学习 ========================
    print()
    print("=" * 60)
    print("阶段3: PPO 强化学习 (RLHF)")
    print("=" * 60)

    # 策略模型 = SFT模型（要优化的对象）
    policy_model = copy.deepcopy(sft_model).to(device)
    # 参考模型 = SFT模型的冻结副本（用于计算KL惩罚）
    ref_model = copy.deepcopy(sft_model).to(device)
    for param in ref_model.parameters():
        param.requires_grad = False
    # 价值模型 = 基于SFT模型初始化
    value_model = ValueModel(copy.deepcopy(sft_model)).to(device)

    ppo_optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-5)   # PPO用更小的学习率
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr=1e-4)

    ppo_dataloader = DataLoader(dummy_ppo_data, batch_size=None)
    num_ppo_rounds = 3
    for round_idx in range(num_ppo_rounds):
        policy_loss, value_loss, entropy = ppo_step(
            policy_model, ref_model, value_model, reward_model,
            ppo_dataloader, ppo_optimizer, value_optimizer, device
        )
        print(f"  Round {round_idx + 1}/{num_ppo_rounds} | "
              f"Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f} | Entropy: {entropy:.4f}")

    # ======================== 推理对比 ========================
    print()
    print("=" * 60)
    print("推理对比: SFT vs PPO")
    print("=" * 60)

    # 用相同的 instruction 生成 response
    test_input = torch.randint(1, vocab_size, (1, instruction_len))

    sft_output = generate_text(sft_model, test_input, max_length=seq_len, device=device)
    ppo_output = generate_text(policy_model, test_input, max_length=seq_len, device=device)

    print(f"  Input (instruction):     {test_input[0].tolist()[:10]}...")
    print(f"  SFT  model output:       ...{sft_output[0].tolist()[-10:]}")
    print(f"  PPO  model output:       ...{ppo_output[0].tolist()[-10:]}")

    # 用奖励模型对比打分
    with torch.no_grad():
        sft_reward = reward_model(sft_output)
        ppo_reward = reward_model(ppo_output)
    print(f"  SFT  reward score:       {sft_reward.item():.4f}")
    print(f"  PPO  reward score:       {ppo_reward.item():.4f}")