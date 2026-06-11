# MiniGPT — 从零训练一个会写诗的语言模型

> 🎯 这是 LLM（大语言模型）入门的 **Hello World** 项目。
> 你不需要任何机器学习背景，只要会 Python 基础就能看懂。

---

## 目录

- [1. 这是什么？](#1-这是什么)
- [2. 前置知识：LLM 到底是什么？](#2-前置知识llm-到底是什么)
- [3. 这个项目做了什么？](#3-这个项目做了什么)
- [4. 代码一步步拆解](#4-代码一步步拆解)
  - [4.1 数据准备：把文字变成数字](#41-数据准备把文字变成数字)
  - [4.2 训练样本：怎么教模型"预测下一个字"](#42-训练样本怎么教模型预测下一个字)
  - [4.3 模型结构：Transformer 是什么？](#43-模型结构transformer-是什么)
  - [4.4 训练循环：模型怎么学习的？](#44-训练循环模型怎么学习的)
  - [4.5 文本生成：模型怎么"写文章"？](#45-文本生成模型怎么写文章)
- [5. 自己动手跑一遍](#5-自己动手跑一遍)
  - [5.1 环境准备](#51-环境准备)
  - [5.2 开始训练](#52-开始训练)
  - [5.3 用模型生成文字](#53-用模型生成文字)
- [6. 怎么读懂训练输出？](#6-怎么读懂训练输出)
- [7. 下一步可以做什么？](#7-下一步可以做什么)
- [8. 参考资料](#8-参考资料)

---

## 1. 这是什么？

简单说：**用 7 万首唐诗宋词作为教材，从零训练一个能续写古典诗句的小型 GPT 模型。**

- 模型只有 **950 万** 个参数（ChatGPT 有上千亿个）
- 在普通笔记本电脑上就能训练，大约 **1.5 小时**
- 训练完成后，你给它一个开头（比如"春风""明月""长安"），它就能续写出古诗风格的句子

它的定位就像编程入门时的 `printf("Hello World")` —— 麻雀虽小，五脏俱全。通过这个项目，你会亲手跑通 LLM 的完整流程：**数据处理 → 模型搭建 → 训练 → 生成**。

### 实际训练效果

在 Apple M2 MacBook 上用 CPU 训练 5 轮的结果：

| 轮次 | Loss | PPL | 说明 |
|------|------|-----|------|
| 1 | 5.00 | 148 | 瞎猜阶段——每字从 148 个候选里挑 |
| 2 | 4.25 | 70 | 学会了常见字搭配 |
| 3 | 3.99 | 54 | 五言、七言节奏浮现 |
| 4 | 3.85 | 47 | 开始有诗意 |
| 5 | **3.73** | **42** | 能写完整诗句 |

训练中模型写出的句子：

- epoch 1: `太白仙人，锦袍初拥。`
- epoch 3: `像祭，广源无复纳降旗。`
- epoch 5: `飞花，花有数、愁无数。`

---

## 2. 前置知识：LLM 到底是什么？

在深入代码之前，先用最朴素的方式解释几个核心概念。

### 2.1 大语言模型（LLM）

> LLM 本质上是一个 **"下一个字预测器"**。

你给它一句话："春风又绿江南"，它猜下一个字最可能是"岸"。给它"床前明月"，它猜下一个字是"光"。

听起来很简单？是的，原理就是这么简单。ChatGPT 能写文章、翻译、编程，本质上都在做同一件事——**根据上文预测下文**，只不过它的规模大了几个数量级。

### 2.2 Token（词元）

计算机不认识汉字，只认识数字。所以我们要把文字切成小块，每个小块叫一个 **token**，然后给每个 token 分配一个数字编号。

比如"床前明月光"可以被切成：

- 按字切（字符级）：`["床", "前", "明", "月", "光"]` → 5 个 token
- 按词切（子词级）：`["床前", "明月", "光"]` → 3 个 token

这个项目用的是**字符级**切分——每个汉字就是一个 token，最简单也最好理解。

### 2.3 训练（Training）

训练就是让模型反复做"完形填空"：

```
输入：床前明月光的疑
标签：前明月光的疑是
```

模型看到"床前明月光的疑"，要预测下一个字是"是"。猜错了？损失函数（loss）会告诉它错得多离谱，然后它调整自己的参数，下次猜得更准。这个过程重复几十万次，模型就慢慢学会了语言的规律。

### 2.4 损失（Loss）和困惑度（Perplexity）

- **Loss（损失）**：模型预测的"错误程度"。越小越好。
  - loss ≈ 8：基本在瞎猜
  - loss ≈ 3-4：已经学到了明显的规律
  - loss ≈ 1-2：预测非常准了

- **Perplexity / PPL（困惑度）**：`math.exp(loss)`，可以理解为"模型在每个位置平均犹豫几个选项"。
  - ppl = 148：每个字要从 148 个候选里选，基本是蒙的
  - ppl = 42：每个字只要从 42 个候选中选，已经很有把握了

### 2.5 Transformer

Transformer 是当代所有 LLM 的底层架构（GPT 中的 T 就是 Transformer）。它最核心的机制叫**自注意力（Self-Attention）**：

> 读到"举头望明___"时，模型通过注意力机制"回顾"前文，发现"望"和"明"离得很近、且"举头"暗示向上看，于是预测空格处填"月"。

你可以粗略地理解为：注意力机制让模型能**在读到每个字时，自动找到前文中最相关的信息**。

---

## 3. 这个项目做了什么？

一张图概括整个流程：

```
chinese-poetry 仓库（JSON，繁体中文）
        │
        ▼  prepare_poetry.py
  下载 + 繁体→简体（opencc）+ 去缺字
        │
        ▼  datas/corpus_poetry.txt（650 万字简体古诗）
        │
        ▼  CharTokenizer
  字符级分词器（9230 个不同汉字 → 0~9229 的 ID）
        │
        ▼  get_batch()
  随机采样训练样本（输入前 128 个字，预测下一个字）
        │
        ▼  MiniGPT
  6 层 Transformer（950 万参数）
        │
        ▼  AdamW 优化器 + 交叉熵损失
  训练 5 轮
        │
        ▼  minigpt.pt（36 MB）
  generate.py → 输入 prompt，续写诗句
```

**数据规模：**

| 指标 | 值 |
|------|-----|
| 数据源 | 全唐诗 5 万首 + 宋词 2.1 万首 + 宋词三百首 |
| 古诗总字数 | 约 650 万字 |
| 不重复的汉字数 | 9,230 个 |
| 模型参数量 | 950 万 |
| 训练时间（Apple Silicon Mac / CPU） | 约 1.5 小时 |

---

## 4. 代码一步步拆解

整个项目的核心代码都在 `train.py` 这一个文件里，大约 220 行。我们逐段拆解。

### 4.1 数据准备：把文字变成数字

#### 4.1.1 读入文本

```python
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text
```

做的事：打开语料文件，把整个文件读成一个字符串。然后把 Windows 风格的换行符 `\r\n` 统一换成 Unix 风格的 `\n`。

语料长这样：

```
静夜思 — 李白
床前明月光，疑是地上霜。
举头望明月，低头思故乡。

春晓 — 孟浩然
春眠不觉晓，处处闻啼鸟。
夜来风雨声，花落知多少。
```

#### 4.1.2 字符级分词器

```python
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))          # 统计所有不重复字符
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # 字 → ID
        self.itos = {i: ch for i, ch in enumerate(chars)}  # ID → 字
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[ch] for ch in s if ch in self.stoi]  # 文字 → 数字序列

    def decode(self, ids):
        return ''.join([self.itos.get(i, '?') for i in ids])  # 数字序列 → 文字
```

这是最简版本的分词器。以"床前明月光"为例：

```python
tokenizer.encode("床前明月光")  # → [1234, 5678, 2345, 6789, 3456]（五个数字）
tokenizer.decode([1234, 5678, 2345, 6789, 3456])  # → "床前明月光"
```

> 💡 **为什么用字符级？** 
> 英文常用子词分词（如 BPE），但中文一个汉字本身就有独立语义。尤其古典诗歌——"春风又绿江南岸"，每个字都不可拆分。字符级分词既简单又精准。7 万首唐诗宋词，不重复的汉字只有 9,230 个——这个"词汇表"对于模型来说非常小。

#### 4.1.3 把整本书编码

```python
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
```

执行后，`data` 就是一个长度为约 650 万的一维张量（你可以理解为数组），每个元素是一个 0~9229 之间的整数。

```
原文："床前明月光，疑是地上霜。"
编码：[1234, 5678, 2345, ...]    ← 650 万个整数排成一列
```

---

### 4.2 训练样本：怎么教模型"预测下一个字"

```python
def get_batch(data, block_size, batch_size):
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y
```

这个函数干了什么？它从全书中**随机抽取** 32 段（`batch_size=32`），每段 128 个字（`block_size=128`）。

关键点：**y 是 x 整体右移一位**。举个例子（假设 block_size=5）：

```
原书位置:  ... 床 前 明 月 光 疑 是 ...
                  │
        x = ["床", "前", "明", "月", "光"]   ← 模型看到的输入
        y = ["前", "明", "月", "光", "疑"]   ← 模型要预测的目标（每个位置都是"下一个字"）
```

这就是语言模型训练的**标准范式**：给定前文，预测下一个字。不需要人工标注，文本本身就是标签。

> 💡 **为什么 batch_size=32？**
> 一次处理 32 个样本而不是 1 个，能让梯度计算更稳定，训练更快。batch_size 越大越稳定，但也越吃内存。32 是适合笔记本的值。

---

### 4.3 模型结构：Transformer 是什么？

这是整个项目的核心。模型由三部分拼接而成：

```
输入 ID 序列（B, T）
     │
     ├──→ Token Embedding（把 ID 变成 256 维向量）
     ├──→ Position Embedding（告诉模型每个字在哪个位置）
     │
     ▼
  相加 → 6 层 TransformerBlock → LayerNorm → Linear → 输出 logits
```

#### 4.3.1 嵌入层（Embedding）

```python
self.token_emb = nn.Embedding(vocab_size, d_model)  # 9230 → 256 维
self.pos_emb = nn.Embedding(block_size, d_model)     # 128 个位置 → 256 维
```

- **Token Embedding**：把每个字的 ID（整数）映射为一个 256 维的向量。这个向量可以理解为"这个字的含义"。
- **Position Embedding**：把位置（第 0 个字、第 1 个字...）也映射为 256 维向量。因为 Transformer 本身不关心顺序，需要显式告诉它"第几个字"的信息。

两个向量**相加**后送入 Transformer：

```python
x = tok_emb + pos_emb   # (32, 128, 256) — 32个样本，每个128字，每字256维
```

#### 4.3.2 TransformerBlock（Decoder 块）

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        # 1. 多头自注意力（Multi-head Self-Attention）
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # 2. 前馈网络（Feed-Forward Network）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # 256 → 1024（放大 4 倍）
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),  # 1024 → 256（缩回来）
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Causal Mask：保证每个字只能看到它前面的字，不能"偷看"后面的
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        # 注意力：让每个字去"查看"前文中所有相关的字
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        # 残差连接 + LayerNorm
        x = self.ln1(x + attn_out)
        # FFN + 残差连接 + LayerNorm
        x = self.ln2(x + self.ffn(x))
        return x
```

一个 TransformerBlock 做两件事：

1. **自注意力（Self-Attention）**：读到"举头望明___"时，"举头""望"和"明"通过注意力机制关联到"月"，于是模型知道这里该填"月"。**Causal Mask** 确保预测第 N 个字时，只能看到前 N-1 个字——不能作弊。
2. **前馈网络（FFN）**：把每个字的表示放大（256→1024），做非线性变换（GELU 激活函数），再缩回来（1024→256）。这步让模型能学到更复杂的语言模式——比如五言、七言的格律规律，对仗平仄的结构。

每层外面都有**残差连接**（`x + attn_out`、`x + ffn(x)`）——简单说就是把输入原样加回去，防止深层网络"学不动"。

这个项目用了 **6 层** 这样的块堆叠起来：

```python
self.blocks = nn.ModuleList([
    TransformerBlock(d_model=256, n_heads=8)
    for _ in range(6)
])
```

#### 4.3.3 输出头（LM Head）

```python
self.lm_head = nn.Linear(d_model, vocab_size)  # 256 → 9230
```

最后一层线性变换，把 256 维的向量映射回 9230 维——每个维度代表词汇表中一个字的"得分"。**得分最高的字就是模型的首选预测。**

#### 4.3.4 关键超参数一览

| 参数 | 值 | 含义 |
|------|-----|------|
| `vocab_size` | 9230 | 词汇表大小（唐诗宋词不重复汉字数） |
| `d_model` | 256 | 每个字的"表示维度"，越大模型越强但越慢 |
| `n_heads` | 8 | 注意力头数，多头 = 从多个角度同时关注 |
| `n_layers` | 6 | Transformer 层数，越深理解能力越强 |
| `block_size` | 128 | 上下文窗口，模型一次最多"看"多少个字 |
| `dropout` | 0.1 | 随机丢弃 10% 的神经元，防止过拟合 |

---

### 4.4 训练循环：模型怎么学习的？

```python
def train(model, data, tokenizer, epochs=5, batch_size=32,
          block_size=128, lr=3e-4, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    steps_per_epoch = len(data) // (batch_size * block_size)
```

#### 4.4.1 训练循环的核心步骤

```python
for epoch in range(epochs):                    # 把全部古诗过 5 遍
    for step in range(steps_per_epoch):         # 每遍约 1584 步
        x, y = get_batch(data, block_size, batch_size)  # ① 随机取一批数据
        logits = model(x)                       # ② 模型前向计算（预测）
        loss = F.cross_entropy(logits, y)       # ③ 算损失（预测和真实差多少）
        optimizer.zero_grad()                   # ④ 清空上一次的梯度
        loss.backward()                         # ⑤ 反向传播（算梯度）
        clip_grad_norm_(model.parameters(), 1.0)# ⑥ 梯度裁剪（防爆炸）
        optimizer.step()                        # ⑦ 更新参数（朝"更准"的方向微调）
```

每一步的含义：

| 步骤 | 做什么 | 通俗理解 |
|------|--------|----------|
| ① | 随机取 32×128 个字 | 翻开诗集，随便找 32 个段落 |
| ② | 模型读输入，输出预测 | "我猜下一个字是…" |
| ③ | 交叉熵损失 | "你猜错了 60%！" |
| ④ | 清空梯度 | 把上一轮的"改正建议"抹掉 |
| ⑤ | 反向传播 | 算出每个参数该往哪个方向改、改多少 |
| ⑥ | 梯度裁剪 | 防止某一步"改太猛"把模型改崩 |
| ⑦ | 更新参数 | 朝正确的方向微调所有 950 万个参数 |

> 💡 **什么是 Epoch？**
> 1 个 epoch = 把 650 万字全部"看过"一遍。5 个 epoch 就是把诗集反复看了 5 遍。每遍模型的理解都会加深一些。

#### 4.4.2 优化器：AdamW

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

- **AdamW** 是 Adam 的改进版，目前最常用的神经网络优化器
- **lr（Learning Rate，学习率）= 0.0003**：控制每一步参数更新的幅度。太大→震荡不收敛，太小→学得太慢

#### 4.4.3 损失函数：交叉熵

```python
loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
```

交叉熵（Cross-Entropy）是分类问题的标配损失函数。这里等价于：

> 模型预测了 9230 个字的概率分布，真实答案是第 K 个字。模型给第 K 个字的概率越高，loss 越小。

---

### 4.5 文本生成：模型怎么"写文章"？

```python
@torch.no_grad()  # 生成时不需要计算梯度，省内存
def generate(self, idx, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):              # 循环生成，一次产生一个字
        idx_cond = idx[:, -self.block_size:]      # 只保留最后 128 个字作为上文
        logits = self(idx_cond)                   # 模型预测下一个字的得分
        logits = logits[:, -1, :] / temperature   # 取最后一个位置的预测，除以温度
        probs = F.softmax(logits, dim=-1)         # 得分 → 概率
        idx_next = torch.multinomial(probs, 1)    # 按概率随机采样
        idx = torch.cat((idx, idx_next), dim=1)   # 把新字拼到序列末尾
    return idx
```

**自回归生成（Autoregressive Generation）**：每次只生成一个字，然后把它拼回原文，再生成下一个字。像推多米诺骨牌。

#### temperature（温度）参数

| temperature | 效果 |
|-------------|------|
| 0.1 ~ 0.5 | 保守，只选高分字，生成内容较重复、较工整 |
| **0.8（默认）** | 平衡，有一定随机性但不离谱 |
| 1.2 ~ 2.0 | 狂野，低分字也有机会被选，更有创意但也更可能前言不搭后语 |

> 原理：`logits / temperature`——温度越低，高分字的优势被放大（更确定）；温度越高，分布被"抹平"，低分字也有机会被选中。

---

## 5. 自己动手跑一遍

### 5.1 环境准备

```bash
# 安装依赖
pip install torch opencc
```

### 5.2 开始训练

```bash
# 1. 下载唐诗宋词数据（繁→简，清洗）
python3 prepare_poetry.py

# 2. 训练模型（CPU 稳定模式，约 1.5 小时）
python3 train.py --cpu
```

训练输出大致长这样：

```
语料长度: 6,490,606 字符
词汇表大小: 9230
模型参数: 9.5M
训练设备: cpu
每轮步数: 1584

  step    0/1584  loss = 9.2016
  step  100/1584  loss = 5.9374
  step  200/1584  loss = 5.6660
  ...

epoch  1/5  loss = 4.9998  ppl = 148.4
  [生成样本] 太白仙人，锦袍初拥。

epoch  2/5  loss = 4.2476  ppl = 69.9
epoch  3/5  loss = 3.9917  ppl = 54.1
  [生成样本] 像祭，广源无复纳降旗。

epoch  4/5  loss = 3.8457  ppl = 46.8
epoch  5/5  loss = 3.7304  ppl = 41.7
  [生成样本] 飞花，花有数、愁无数。

模型已保存到 minigpt.pt
```

### 5.3 用模型生成文字

```bash
python3 generate.py
```

```
MiniGPT 已加载 | 参数: 9.5M | 设备: cpu
词表: 9230 | 上下文: 128
输入 prompt 开始生成（输入 'quit' 退出）

Prompt > 春风
春风不解意，吹梦到西洲。
Prompt > 明月
明月几时有，把酒问青天。
Prompt > quit
```

---

## 6. 怎么读懂训练输出？

| 输出 | 含义 |
|------|------|
| `step 0/1584` | 当前 epoch 的第几步 / 总步数 |
| `loss = 5.93` | 当前这批数据的预测损失（越低越好） |
| `epoch 1/5` | 第几个 epoch 完成 / 总 epoch 数（这里显示的是 epoch 完成时的平均 loss） |
| `ppl = 148.4` | 困惑度，模型在每个位置平均犹豫几个字（越低越好） |
| `[生成样本]` | 训练中途随机取一段上下文，让模型续写的文字 |

**loss 下降的正常节奏：**

| loss 范围 | 模型状态 |
|-----------|----------|
| 8 ~ 9 | 随机初始化，纯瞎猜 |
| 5 ~ 7 | 学会了高频字（的、是、不、人） |
| 3 ~ 5 | 掌握了常见搭配和节奏 |
| < 3 | 学得很好了，能写通顺句子 |

---

## 7. 下一步可以做什么？

训练完成后，你可以尝试这些方向：

- **换诗人专项训练**：从语料中筛出李白、杜甫、苏轼的诗单独训练，得到某个诗人的"仿写模型"
- **调大模型**：把 `d_model` 从 256 改到 384，`n_layers` 从 6 改到 8（但要防备 M2 Air 过热降频）
- **加 BPE 分词器**：替换字符级为子词级（`tokenizers` 库），压缩率更好但需要额外训练分词器
- **用 MPS 加速**：去掉 `--cpu` 参数，在 Apple GPU 上跑（快 3 倍但可能卡死，建议空调房）
- **加温度调节**：`python3 generate.py --temperature 1.2` 看更大胆的创作

---

## 8. 参考资料

- [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) — 最全的中华古典诗词开源数据集
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- [karpathy/minGPT](https://github.com/karpathy/minGPT) — 本项目架构的灵感来源
- [The Roads for LLM](https://mcell.top/books/the-roads-for-llm/train-small-gpt) — 配套中文教程
