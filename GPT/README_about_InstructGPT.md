 # InstructGPT 实现

 ## InstructGPT 三阶段

 InstructGPT 的核心是 **RLHF（基于人类反馈的强化学习）**，分为三个阶段：

 1. SFT（监督微调） — 用人类标注的 instruction-response 对微调预训练 GPT
 2. RM（奖励模型） — 用人类偏好数据（response 排序）训练一个打分模型
 3. PPO（近端策略优化） — 用 RM 的奖励信号通过强化学习优化策略模型

 ## 实现方案

 ### 文件结构

 单文件 GPT/instructgpt.py，包含以下类和函数：

 #### 共享组件（复用 gpt1/2/3 的风格）

 - MultiHeadAttention — 与 GPT-2 一致（GELU + Dropout + Pre-Norm），去掉稀疏注意力

     注：InstructGPT 用的是正常的（稠密）注意力机制，没有使用稀疏注意力。它直接沿用了 GPT-3 的基础模型架构，而 **GPT-3 的注意力机制并非在所有规模下都是稀疏的，是在大模型（如13B、175B版本）的部分层才使用了稀疏注意力来降低计算量，而小模型仍用正常的稠密注意力**。InstructGPT 作为基于 GPT-3 微调的模型，自然也就继承了这一架构设定，其核心创新在于通过指令微调（Supervised Fine-Tuning）和人类反馈强化学习（RLHF）来提升模型遵循指令的能力。

 - PositionwiseFeedForward — 与 GPT-2 一致
 - DecoderLayer — Pre-Norm，与 GPT-2/3 一致

 #### 模型类

 1. GPTModel — 基础 GPT 模型（token embed + pos embed + N 层 decoder + fc），SFT 阶段的策略模型
 2. RewardModel — 继承 GPTModel 的 transformer 部分，去掉 fc 层，加一个 nn.Linear(d_model, 1) 输出标量奖励分数
 3. ValueModel — 与 RewardModel 类似，输出每个 token 位置的状态价值（用于 PPO 的 baseline）

 ### 训练函数

 1. **train_sft() — SFT 阶段**，标准的监督微调（instruction + response 拼接，计算 response 部分的交叉熵损失）
 2. **train_reward_model() — RM 阶段**，输入两个 response 的得分，用 Bradley-Terry 模型计算偏好损失：loss = -log(sigmoid(r_chosen - r_rejected))
 3. **ppo_step() — PPO 阶段**：
   - 用策略模型生成 response
   - 用 RM 打分得到奖励 R
   - 用 Value Model 估计当前状态的期望奖励（baseline）
   - 计算 GAE（广义优势估计）

     baseline 的作用是降低方差、稳定训练。Value Model 预估当前状态的"期望奖励" V(s)，
     然后计算优势函数 A = R - V(s)。如果 A > 0，说明这个 action 比预期好，增大概率；
     如果 A < 0，说明比预期差，减小概率。

   - PPO-Clip 目标函数更新策略

     PPO-Clip 的核心思想是限制策略更新幅度，防止一次更新太大导致训练崩溃。定义新旧策略的比率 r(θ) = π_new(a|s) / π_old(a|s)，目标函数为：
     L = min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)，其中 ε 通常取 0.2。
     当 A > 0（好的 action）时，想增大概率，但 r(θ) 被限制在 [1-ε, 1+ε] 之间，
     不会一下子变得太大；当 A < 0（差的 action）时，想减小概率，同样被 clip 限制。这样每次更新都被夹在一个安全范围内，训练更稳定。

   - 更新 Value Model

 ### 测试数据

 与 GPT-1/2/3 保持一致的风格：
 - vocab_size = 10000
 - d_model = 128
 - num_heads = 4
 - num_layers = 2
 - max_seq_length = 512
 - d_ff = 512

 Dummy 数据模拟 instruction-response 对和偏好对比数据。

 ### main 流程

 #### 阶段1: SFT 监督微调
 sft_model = train_sft(...)

 #### 阶段2: 训练奖励模型
 reward_model = train_reward_model(...)

 #### 阶段3: PPO 强化学习
 ppo_model = ppo_step(policy_model, ref_model, value_model, reward_model, ...)

 ### 验证方式

 运行 python instructgpt.py，观察三个阶段的 loss 收敛情况。

 三个阶段全部运行成功，loss 正常收敛：

  - SFT: 9.26 → 9.01（loss 下降）
  
  - RM: 0.83 → 0.75（偏好学习收敛）

  - PPO: Policy Loss 和 Value Loss 都在下降，PPO 模型的奖励分数略高于 SFT 模型
  
## 总结

  GPT/instructgpt.py 包含 InstructGPT 的三个阶段：

  1. SFT 监督微调 — 在 instruction-response 对上微调 GPT，只计算 response 部分的交叉熵损失
  2. RM 奖励模型 — 用 Bradley-Terry 偏好模型训练（-log(sigmoid(r_chosen - r_rejected))）
  3. PPO 强化学习 — 包含 KL 惩罚（防止偏离参考模型）、GAE 优势估计、PPO-Clip 目标函数、熵正则化（鼓励策略保持探索）
                                                                                                    
  最后还有一个 SFT vs PPO 的推理对比，展示 PPO 模型获得了更高的奖励分数。