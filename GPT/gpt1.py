import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # softmax((Q * K^T) / sqrt(d_k)) * V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)    # 需要mask的地方用一个很大的负数填充，softmax后为0
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)   # [batch_size, num_heads, seq_length, d_k]

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):    # 这里的 QKV 都是 输入x = embeddings = token_embeds + position_embeds
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)   # d_model × 4 = d_ff
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)    # 3个x 分别代表 Q, K, V
        x = self.norm1(x + self.dropout(attn_output))    # add & norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))      # add & norm
        return x


class GPT1(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_length, d_ff, dropout=0.1):
        super(GPT1, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_length, d_model)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)   # num_layers 层 Decoder 堆叠
        ])
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        # 创建位置序列 -> 增加批次维度 -> 复制到批次大小
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(
            input_ids.size(0), 1)
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(positions)
        embeddings = token_embeds + position_embeds
        embeddings = self.dropout(embeddings)
        # torch.tril() - 取矩阵的下三角部分，上三角置0
        mask = torch.tril(torch.ones(seq_length, seq_length, device=input_ids.device)).unsqueeze(0).unsqueeze(0)   # [batch_dim, head_dim, seq_len, seq_len]

        x = embeddings   # text & position embed
        for layer in self.decoder_layers:
            x = layer(x, mask)

        logits = self.fc(x)
        return logits


# 训练
def train_gpt1(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for input_ids in dataloader:
        input_ids = input_ids.to(device)   # [batch_size, seq_length]
        optimizer.zero_grad()
        logits = model(input_ids)   # logits shape: [batch_size, seq_length, vocab_size]
        shift_logits = logits[..., :-1, :].contiguous()   # [batch_size, seq_length[:-1], vocab_size]
        shift_labels = input_ids[..., 1:].contiguous()    # [batch_size, seq_length[1:]]  用位置i的logits预测位置i+1的token
        # 交叉熵损失
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),   # [batch_size*(seq_len-1), vocab_size]  把批次和序列维度合并，每个token预测变成一个独立的预测任务
            shift_labels.view(-1)   # [batch_size*(seq_len-1)]  每个预测位置对应一个标签值
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # 梯度裁剪，防止深度模型训练出现梯度爆炸
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# 推理 (自回归文本生成)
def generate_text(model, input_ids, max_length, device):
    model.eval()
    input_ids = input_ids.to(device)    # shape: [batch_size, current_seq_len]
    with torch.no_grad():
        for _ in range(max_length - input_ids.size(1)):
            logits = model(input_ids)   # logits shape: [batch_size, current_seq_len, vocab_size]
            # 自回归的本质：
            # 给定序列 x1, x2, ..., xt
            # 预测 x_{t+1} = argmax P(x_{t+1} | x1, x2, ..., xt)
            # 最后一个位置的输出正是这个条件概率
            next_token_logits = logits[:, -1, :]   # [batch_size, vocab_size]  取最后一个序列位置的预测 (只关心最后一个位置的下一个token预测)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)    # 在最后一个维度（vocab_size）上找最大值索引  unsqueeze -> [batch_size, 1]
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)    # [batch_size, current_seq_len + 1]
    return input_ids


# 示例参数
vocab_size = 10000
d_model = 128
num_heads = 4
num_layers = 2
max_seq_length = 512
d_ff = 512

#################### GPT-1中的实际参数 ####################
# vocab_size = 40478   # 词汇表大小
# d_model = 768        # 词向量维度
# num_heads = 12       # 12个注意力头
# num_layers = 12      # 12层 Transformer Decoder
# max_seq_length = 512    # 最大序列长度 512 tokens
# d_ff = 3072    # 768 × 4
#################### GPT-1中的实际参数 ####################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型
model = GPT1(vocab_size, d_model, num_heads, num_layers, max_seq_length, d_ff).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)   # GPT-1中的实际参数 lr=2.5e-4

# 模拟数据加载器
batch_size = 16   # GPT-1中的实际参数 batch_size = 64
num_batches = 10
dummy_data = [
    torch.randint(0, vocab_size, (batch_size, max_seq_length))
    for _ in range(num_batches)
]
dataloader = torch.utils.data.DataLoader(dummy_data, batch_size=None)

# 训练模型
num_epochs = 5   # GPT-1中的实际参数 num_epochs = 100
for epoch in range(num_epochs):
    loss = train_gpt1(model, dataloader, criterion, optimizer, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")


# 推理示例
input_text = torch.randint(0, vocab_size, (1, 10))    # 1个批次（batch_size=1），序列长度10个token
generated_text = generate_text(model, input_text, max_length=20, device=device)
print("Generated text:", generated_text)