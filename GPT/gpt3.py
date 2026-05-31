import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SparseMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, local_window=3, stride=4, dropout=0.1):
        super(SparseMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.local_window = local_window  # 局部窗口大小k
        self.stride = stride  # 跨步步长

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def create_strided_mask(self, seq_length, device):
        """位置i只能看到位置j，如果满足：
        1. j < i (自回归，只能看到前面的)
        2. (i - j <= self.local_window) 或者 ((i - j) % self.stride == 0)
        """
        mask = torch.zeros(seq_length, seq_length, device=device)

        for i in range(seq_length):
            # 规则1：局部窗口 - 相对距离不超过k
            start = max(0, i - self.local_window)  # 不能小于0
            mask[i, start:i + 1] = 1  # 位置i可以看到从start到i的所有位置

            # 规则2：跨步连接 - 距离为k, 2k, 3k...
            for step in range(1, seq_length // self.stride + 1):
                distance = step * self.stride
                prev_pos = i - distance  # 计算前面的位置
                if prev_pos >= 0:  # 位置有效（不越界）
                    mask[i, prev_pos] = 1

        mask = torch.tril(mask)  # 下三角掩码（自回归）
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

    def attention(self, Q, K, V, mask=None, strided_mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if strided_mask is not None:  # 稀疏掩码
            scores = scores.masked_fill(strided_mask == 0, -1e9)

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
        seq_length = Q.size(1)
        strided_mask = self.create_strided_mask(seq_length, Q.device)

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.attention(Q, K, V, mask, strided_mask)  # 应用稀疏注意力
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, local_window=3, stride=4, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = SparseMultiHeadAttention(d_model, num_heads, local_window, stride, dropout)
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


class GPT3(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_length, d_ff):
        super(GPT3, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_length, d_model)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self._init_weights()

    def _init_weights(self):
        """对残差输出层应用 1/√N 缩放"""
        scale = 1.0 / math.sqrt(self.num_layers)

        # 对所有线性层使用标准初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)  # 0.02 是 GPT-2/3 中标准权重初始化的标准差
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 然后对残差输出层进行额外缩放
        for layer in self.decoder_layers:
            # 缩放注意力输出投影
            layer.self_attn.W_o.weight.data *= scale
            # 缩放 FFN 输出层
            layer.feed_forward.fc2.weight.data *= scale

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(
            input_ids.size(0), 1)
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(positions)
        embeddings = token_embeds + position_embeds

        mask = torch.tril(torch.ones(seq_length, seq_length, device=input_ids.device)).unsqueeze(0).unsqueeze(0)

        x = embeddings
        for layer in self.decoder_layers:
            x = layer(x, mask)

        logits = self.fc(x)
        return logits


# 训练示例
def train_gpt3(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for input_ids in dataloader:
        input_ids = input_ids.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止深度模型训练出现梯度爆炸
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# 推理示例
def generate_text(model, input_ids, max_length, device):
    model.eval()
    input_ids = input_ids.to(device)
    with torch.no_grad():
        for _ in range(max_length - input_ids.size(1)):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    return input_ids


# 示例参数
vocab_size = 10000
d_model = 128
num_heads = 4
num_layers = 2
max_seq_length = 512
d_ff = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型
model = GPT3(vocab_size, d_model, num_heads, num_layers, max_seq_length, d_ff).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 模拟数据加载器
batch_size = 16
num_batches = 10
dummy_data = [
    torch.randint(0, vocab_size, (batch_size, max_seq_length))
    for _ in range(num_batches)
]
dataloader = torch.utils.data.DataLoader(dummy_data, batch_size=None)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    loss = train_gpt3(model, dataloader, criterion, optimizer, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

# 推理示例
input_text = torch.randint(0, vocab_size, (1, 10))
generated_text = generate_text(model, input_text, max_length=20, device=device)
print("Generated text:", generated_text)