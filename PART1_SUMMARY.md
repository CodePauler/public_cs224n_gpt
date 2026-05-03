# CS 224N Default Final Project — Part 1 总结

## 项目概述

Part 1 的目标是从零实现 GPT-2（small, 124M 参数）的核心组件，包括多头因果自注意力机制、Pre-LN Transformer 层、完整 GPT-2 模型、情感分类头以及 AdamW 优化器。最终在 Stanford Sentiment Treebank（SST, 5 类情感）和 Cornell Film IMDb（CFIMDB, 二分类情感）两个数据集上进行情感分类训练与评估。

---

## GPT-2 模型架构

### 整体结构

```
输入 Token IDs [B, T]
  │
  ▼
Word Embedding + Position Embedding + Dropout    (embed)
  │
  ▼
[×12  GPT2Layer Block]:                          (encode)
  │  LayerNorm → CausalSelfAttention → Linear → Dropout → +Residual
  │  LayerNorm → Linear → GELU → Linear → Dropout → +Residual
  │
  ▼
Final LayerNorm                                  (final_layer_norm)
  │
  ▼
Hidden States [B, T, 768]
  │
  └── 取最后非 padding token → Dropout → Linear → Logits [num_labels]
```

### 关键设计

| 特性 | 说明 |
|------|------|
| **Pre-LN** | LayerNorm 放在每个子层 *之前*（原始 Transformer 放在 *之后*），训练更稳定 |
| **Causal Mask** | 上三角矩阵 `-inf` 确保 token 只能关注自身及前面的 token（自回归） |
| **Weight Tying** | 词嵌入权重与输出投影共享：`logits = hidden_state @ word_embedding^T` |
| **GELU 激活** | FFN 层使用 `F.gelu`（Gaussian Error Linear Unit） |
| **BPE Tokenizer** | 使用 GPT-2 原始 tokenizer，vocab_size = 50257，max_position = 1024 |

### 模型参数

| 超参数 | 值 |
|--------|---|
| Hidden Size (`d_model`) | 768 |
| Attention Heads | 12 |
| Head Dim (`d_k`) | 64 |
| Transformer Layers | 12 |
| FFN Intermediate Size | 3072（4× d_model） |
| Vocab Size | 50257 |
| Max Position | 1024 |
| Hidden Dropout | 0.1 |
| Attention Dropout | 0.1 |
| LayerNorm Epsilon | 1e-5 |
| 总参数量 | ≈124M |

### CausalSelfAttention 详细结构

1. **Q/K/V 投影**：`Linear(768, 768)` 各一个 → 从 `[B, T, D]` reshape 为 `[B, H, T, Dh]`
2. **注意力计算**：`scores = QK^T / sqrt(64)` → 加 causal mask → 加 padding mask → softmax → dropout
3. **因果掩码**：`torch.triu(ones(T, T), diagonal=1)` 上三角置 `-inf`
4. **输出合并**：`[B, H, T, Dh]` → rearranged 为 `[B, T, D]`

### GPT2Layer 数据流

```
hidden_states [B, T, 768]
  → LayerNorm
  → CausalSelfAttention
  → Linear(768→768) + Dropout(0.1)
  → + hidden_states (残差连接)
  → LayerNorm
  → Linear(768→3072) + GELU
  → Linear(3072→768) + Dropout(0.1)
  → + 残差连接
  → output [B, T, 768]
```

---

## 实现清单

Part 1 在以下 5 个文件中完成代码实现（填充 `### YOUR CODE HERE` 占位符）：

| 文件 | 实现内容 | 状态 |
|------|---------|------|
| `modules/attention.py` | `CausalSelfAttention.transform()`, `.attention()`, `.forward()` | 已完成 |
| `modules/gpt2_layer.py` | `GPT2Layer.add()`, `.forward()` | 已完成 |
| `models/gpt2.py` | `GPT2Model.__init__()`, `.from_pretrained()`, `.embed()`, `.encode()`, `.forward()` | 已完成 |
| `classifier.py` | `GPT2SentimentClassifier.__init__()`, `.forward()` | 已完成 |
| `optimizer.py` | `AdamW.__init__()`, `.step()` | 已完成 |

**验证工具**：
- `sanity_check.py` — 对比自定义 GPT2Model 与 Hugging Face 官方模型的输出，确保实现正确（误差 < 0.1）
- `optimizer_test.py` — 在简单线性回归任务上运行 1000 步 AdamW，与参考张量比较（atol=1e-6）

---

## AdamW 优化器

| 参数 | 默认值 |
|------|--------|
| 学习率 (`lr`) | 1e-3 |
| Betas (`β₁, β₂`) | (0.9, 0.999) |
| Epsilon (`ε`) | 1e-6 |
| Weight Decay | 0.0 |

**算法要点**：
1. 维护每个参数的指数移动平均（一阶矩 `exp_avg` 和二阶矩 `exp_avg_sq`）
2. 使用高效版偏差校正：`step_size = lr * sqrt(1 - β₂^t) / (1 - β₁^t)`
3. **Weight decay 与梯度更新解耦**（AdamW vs Adam 的关键区别）：`p -= lr * weight_decay * p` 在动量更新之后单独执行

---

## 数据集

### SST（Stanford Sentiment Treebank）— 5 分类

| 数据集 | 样本数 | 有标签？ |
|--------|--------|---------|
| `ids-sst-train.csv` | 8,544 | 是 |
| `ids-sst-dev.csv` | 1,101 | 是 |
| `ids-sst-test-student.csv` | 2,210 | 否（提交用） |

**标签分布（SST Train）**：0=最负面(1092), 1(2218), 2=中性(1624), 3(2322), 4=最正面(1288)

### CFIMDB（Cornell Film IMDb）— 二分类

| 数据集 | 样本数 | 有标签？ |
|--------|--------|---------|
| `ids-cfimdb-train.csv` | 1,707 | 是 |
| `ids-cfimdb-dev.csv` | 245 | 是 |
| `ids-cfimdb-test-student.csv` | 488 | 否（提交用） |

**标签分布（CFIMDB Train）**：0=负面(856), 1=正面(851)

---

## 训练过程

### 情感分类器结构

`GPT2SentimentClassifier` 在预训练 GPT-2 基础上，取**最后一个非 padding token** 的 768 维隐藏状态，经过 Dropout + Linear 分类头输出 logits。

### 两种微调模式

| 模式 | 训练参数 | 说明 |
|------|---------|------|
| `last-linear-layer` | 仅分类头 + Dropout | GPT-2 所有参数冻结，只训练最后一层 |
| `full-model` | 所有参数 | GPT-2 全部解冻，端到端微调 |

### 训练超参数

| 参数 | 值 |
|------|---|
| 轮数 (Epochs) | 10 |
| 随机种子 | 11711 |
| 分类器 Dropout | 0.3 |
| 损失函数 | Cross Entropy Loss |
| SST Batch Size | 建议 64（默认 8） |
| CFIMDB Batch Size | 8 |

### 训练流程

1. 用 `GPT2Tokenizer` 对文本编码（`eos_token` 作为 `pad_token`）
2. 每个 epoch：`model.train()` → 前向 → `cross_entropy` → 反向传播 → AdamW 更新
3. 每轮结束后在 train/dev 集上评估 accuracy 和 macro F1
4. 保存 dev accuracy 最高的模型 checkpoint（含 model_config, optimizer state, 随机状态）

---

## 实验结果

### 官方 Baseline（Dev Set）

| 设置 | SST Dev Accuracy | CFIMDB Dev Accuracy |
|------|-----------------|--------------------|
| Last Linear Layer | 0.462 | 0.861 |
| Full Model | 0.513 | 0.976 |

### 我的训练结果（Dev Set）

| 设置 | SST Dev Accuracy | CFIMDB Dev Accuracy |
|------|-----------------|--------------------|
| Last Linear Layer | 0.462 | 0.882 |
| Full Model | 0.506 | 0.976 |

---

## 文件结构

```
├── models/gpt2.py           # GPT-2 完整模型（embed + encode + weight tying）
├── modules/attention.py     # 多头因果自注意力
├── modules/gpt2_layer.py    # Pre-LN GPT-2 Transformer 层
├── classifier.py            # 情感分类训练/评估
├── optimizer.py             # AdamW 优化器实现
├── sanity_check.py          # GPT-2 实现正确性验证
├── optimizer_test.py        # AdamW 正确性验证
├── config.py                # 配置类（PretrainedConfig, GPT2Config）
├── utils.py                 # 工具函数（mask 生成、模型下载缓存）
├── datasets.py              # 数据集加载（para+phrase + sonnets, Part 2）
├── evaluation.py            # 评估工具
├── paraphrase_detection.py  # 释义检测（Part 2）
├── sonnet_generation.py     # 十四行诗生成（Part 2）
└── data/                    # 所有数据集
```
