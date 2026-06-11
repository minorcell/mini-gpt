# MiniGPT — 从零训练一个"西游记"语言模型

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

简单说：**用《西游记》全文作为教材，从零训练一个能写"西游记风格"文字的小型 GPT 模型。**

- 模型只有 **710 万** 个参数（ChatGPT 有上千亿个）
- 在普通笔记本电脑上就能训练，大约 **10-20 分钟**
- 训练完成后，你给它一个开头（比如"行者"），它就能续写出西游记风格的段落

它的定位就像编程入门时的 `printf("Hello World")` —— 麻雀虽小，五脏俱全。通过这个项目，你会亲手跑通 LLM 的完整流程：**数据处理 → 模型搭建 → 训练 → 生成**。

---

## 2. 前置知识：LLM 到底是什么？

在深入代码之前，先用最朴素的方式解释几个核心概念。

### 2.1 大语言模型（LLM）

> LLM 本质上是一个 **"下一个字预测器"**。

你给它一句话："今天天气真"，它猜下一个字最可能是"好"。给它"孙悟空举起"，它猜下一个字是"金箍棒"。

听起来很简单？是的，原理就是这么简单。ChatGPT 能写文章、翻译、编程，本质上都在做同一件事——**根据上文预测下文**，只不过它的规模大了几个数量级。

### 2.2 Token（词元）

计算机不认识汉字，只认识数字。所以我们要把文字切成小块，每个小块叫一个 **token**，然后给每个 token 分配一个数字编号。

比如"我是孙悟空"可以被切成：

- 按字切（字符级）：`["我", "是", "孙", "悟", "空"]` → 5 个 token
- 按词切（子词级）：`["我", "是", "孙悟空"]` → 3 个 token

这个项目用的是**字符级**切分——每个汉字就是一个 token，最简单也最好理解。

### 2.3 训练（Training）

训练就是让模型反复做"完形填空"：

```
输入：唐僧师徒四人去西天
标签：僧师徒四人去西天取
```

模型看到"唐僧师徒四人去西天"，要预测下一个字是"取"。猜错了？损失函数（loss）会告诉它错得多离谱，然后它调整自己的参数，下次猜得更准。这个过程重复几十万次，模型就慢慢学会了语言的规律。

### 2.4 损失（Loss）和困惑度（Perplexity）

- **Loss（损失）**：模型预测的"错误程度"。越小越好。
  - loss ≈ 8：基本在瞎猜
  - loss ≈ 3-4：已经学到了明显的规律
  - loss ≈ 1-2：预测非常准了

- **Perplexity / PPL（困惑度）**：`math.exp(loss)`，可以理解为"模型在每个位置平均犹豫几个选项"。
  - ppl = 295：每个字要从 295 个候选里选，基本是蒙的
  - ppl = 31：每个字只要从 31 个候选中选，已经很有把握了

### 2.5 Transformer

Transformer 是当代所有 LLM 的底层架构（GPT 中的 T 就是 Transformer）。它最核心的机制叫**自注意力（Self-Attention）**：

> 读到"孙悟空举起___"时，模型通过注意力机制"回顾"前文，发现"金箍棒"和"孙悟空"高度相关，于是预测空格处填"金箍棒"。

你可以粗略地理解为：注意力机制让模型能**在读到每个字时，自动找到前文中最相关的信息**。

---

## 3. 这个项目做了什么？

一张图概括整个流程：

```
《西游记》文本（GB18030 编码）
        │
        ▼
   UTF-8 纯文本（corpus.txt）
        │
        ▼
  字符级分词器（4529 个不同汉字 → 0~4528 的 ID）
        │
        ▼
  随机采样训练样本（输入前 128 个字，预测下一个字）
        │
        ▼
  MiniGPT 模型（6 层 Transformer，710 万参数）
        │
        ▼
  AdamW 优化器 + 交叉熵损失 → 训练 10 轮
        │
        ▼
  训练好的模型（minigpt.pt）→ 可以生成西游记风格文字
```

**数据规模：**

| 指标 | 值 |
|------|-----|
| 《西游记》全文字数 | 约 73 万字 |
| 不重复的汉字数 | 4,529 个 |
| 模型参数量 | 710 万 |
| 训练时间（Apple Silicon Mac） | 约 10-15 分钟 |

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

做的事：打开 `corpus.txt`，把整个文件读成一个字符串。然后把 Windows 风格的换行符 `\r\n` 统一换成 Unix 风格的 `\n`。

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

这是最简版本的分词器。以"孙悟空"为例：

```python
tokenizer.encode("孙悟空")  # → [3812, 2589, 1060]（三个数字代表三个字）
tokenizer.decode([3812, 2589, 1060])  # → "孙悟空"
```

> 💡 **为什么用字符级？** 
> 英文常用子词分词（如 BPE），但中文一个汉字本身就有独立语义，字符级分词既简单又管用。《西游记》全书 73 万字，不重复的汉字只有 4,529 个——这个"词汇表"对于模型来说非常小。

#### 4.1.3 把整本书编码

```python
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
```

执行后，`data` 就是一个长度为 731,625 的一维张量（你可以理解为数组），每个元素是一个 0~4528 之间的整数。

```
原书："第一回 灵根育孕源流出 心性修持大道生" 
编码：[1584, 21, 1722, ...]    ← 73 万个整数排成一列
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
原书位置:  ... 唐 僧 师 徒 四 人 去 ...
                  │
        x = ["唐", "僧", "师", "徒", "四"]   ← 模型看到的输入
        y = ["僧", "师", "徒", "四", "人"]   ← 模型要预测的目标（每个位置都是"下一个字"）
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
self.token_emb = nn.Embedding(vocab_size, d_model)  # 4529 → 256 维
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

1. **自注意力（Self-Attention）**：读到"孙悟空举起__"时，"孙悟空"和"举起"通过注意力机制关联到"金箍棒"，于是模型知道这里该填武器名。**Causal Mask** 确保预测第 N 个字时，只能看到前 N-1 个字——不能作弊。
2. **前馈网络（FFN）**：把每个字的表示放大（256→1024），做非线性变换（GELU 激活函数），再缩回来（1024→256）。这步让模型能学到更复杂的语言模式。

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
self.lm_head = nn.Linear(d_model, vocab_size)  # 256 → 4529
```

最后一层线性变换，把 256 维的向量映射回 4529 维——每个维度代表词汇表中一个字的"得分"。**得分最高的字就是模型的首选预测。**

#### 4.3.4 关键超参数一览

| 参数 | 值 | 含义 |
|------|-----|------|
| `vocab_size` | 4529 | 词汇表大小（《西游记》不重复汉字数） |
| `d_model` | 256 | 每个字的"表示维度"，越大模型越强但越慢 |
| `n_heads` | 8 | 注意力头数，多头 = 从多个角度同时关注 |
| `n_layers` | 6 | Transformer 层数，越深理解能力越强 |
| `block_size` | 128 | 上下文窗口，模型一次最多"看"多少个字 |
| `dropout` | 0.1 | 随机丢弃 10% 的神经元，防止过拟合 |

---

### 4.4 训练循环：模型怎么学习的？

```python
def train(model, data, tokenizer, epochs=10, batch_size=32,
          block_size=128, lr=3e-4, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    steps_per_epoch = len(data) // (batch_size * block_size)
```

#### 4.4.1 训练循环的核心步骤

```python
for epoch in range(epochs):                    # 把全书过 10 遍
    for step in range(steps_per_epoch):         # 每遍约 178 步
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
| ① | 随机取 32×128 个字 | 翻开书，随便找 32 个段落 |
| ② | 模型读输入，输出预测 | "我猜下一个字是..." |
| ③ | 交叉熵损失 | "你猜错了 60%！" |
| ④ | 清空梯度 | 把上一轮的"改正建议"抹掉 |
| ⑤ | 反向传播 | 算出每个参数该往哪个方向改、改多少 |
| ⑥ | 梯度裁剪 | 防止某一步"改太猛"把模型改崩 |
| ⑦ | 更新参数 | 朝正确的方向微调所有 710 万个参数 |

> 💡 **什么是 Epoch？**
> 1 个 epoch = 把全书 73 万字全部"看过"一遍。10 个 epoch 就是把书反复看了 10 遍。每遍模型的理解都会加深一些。

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

> 模型预测了 4529 个字的概率分布，真实答案是第 K 个字。模型给第 K 个字的概率越高，loss 越小。

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
| 0.1 ~ 0.5 | 保守，只选高分字，生成内容较重复 |
| **0.8（默认）** | 平衡，有一定随机性但不离谱 |
| 1.2 ~ 2.0 | 狂野，低分字也有机会被选，更有创意但也更可能前言不搭后语 |

> 原理：`logits / temperature`——温度越低，高分字的优势被放大（更确定）；温度越高，分布被"抹平"，低分字也有机会被选中。

---

## 5. 自己动手跑一遍

### 5.1 环境准备

你需要 Python 3.8+ 和 PyTorch。

```bash
# 1. 克隆项目（或直接进入项目目录）
cd mini-gpt

# 2. 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 3. 安装 PyTorch
pip install torch
```

> **Mac 用户（Apple Silicon）**：PyTorch 会自动使用 MPS 加速，训练只需 10-15 分钟。
> **Windows / Linux 用户**：有 NVIDIA 显卡会自动用 CUDA，没有则用 CPU（会慢一些，大约 30-60 分钟）。

### 5.2 开始训练

```bash
python3 train.py
```

你会看到类似这样的输出：

```
语料长度: 731,625 字符
词汇表大小: 4529
模型参数: 7.1M
训练设备: mps
每轮步数: 178

  step    0/178  loss = 8.4683
  step  100/178  loss = 5.2202
epoch  1/10  loss = 5.6879  ppl = 295.3
  [生成样本] 等老猪看守师父。...
```

### 5.3 用模型生成文字

训练完成后，运行交互式生成脚本：

```bash
python3 generate.py
```

```
MiniGPT 已加载 | 参数: 7.1M | 设备: mps
输入 prompt 开始生成（输入 'quit' 退出）

Prompt > 行者
行者道："我正是大王梦斩内外公主，免得这妖怪..." 

Prompt > 却说
却说真是：仙童吩咐，娘娘娘娘又得出来...

Prompt > quit
```

也可以直接在 Python 里调用：

```python
import torch
from train import MiniGPT, CharTokenizer, load_data

text = load_data("corpus.txt")
tokenizer = CharTokenizer(text)

model = MiniGPT(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load("minigpt.pt", weights_only=True))
model.eval()

# 生成
context = torch.tensor([tokenizer.encode("行者")])
gen = model.generate(context, max_new_tokens=100, temperature=0.8)
print(tokenizer.decode(gen[0].tolist()))
```

---

## 6. 怎么读懂训练输出？

这是本项目的实际训练结果：

```
epoch  1/10  loss = 5.6879  ppl = 295.3   ← 刚开始，基本瞎猜
epoch  2/10  loss = 4.7341  ppl = 113.8
epoch  3/10  loss = 4.3852  ppl =  80.3
epoch  4/10  loss = 4.1577  ppl =  63.9
epoch  5/10  loss = 3.9935  ppl =  54.2   ← 过半，已学到明显规律
epoch  6/10  loss = 3.8573  ppl =  47.3
epoch  7/10  loss = 3.7482  ppl =  42.4
epoch  8/10  loss = 3.6395  ppl =  38.1
epoch  9/10  loss = 3.5301  ppl =  34.1
epoch 10/10  loss = 3.4381  ppl =  31.1   ← 最终：平均从 31 个候选字中选
```

**loss 下降 = 模型在进步。** ppl 从 295 降到 31，意味着模型从"295 个选项里蒙一个"进步到了"31 个选项里就有正确答案"。

生成样本的演变也很直观：

| Epoch | 生成样例 | 质量 |
|-------|----------|------|
| 1 | "等老猪看守师父。"大王道："徒弟们师父公主之二陌。" | 不成句，词语碎片化 |
| 5 | 飞绛绮。更无一缕青烟而入。南无边流星君... | 有古典韵味，但逻辑混乱 |
| 10 | 那马腰软蹄弯，即便跪下，见八戒、沙僧面前拱手道："师兄，你还不一钯..." | 角色正确，情节合理，对话通顺 |

> 💡 **为什么 ppl 到 31 就下不去了？**
> 因为这是一个只有 7M 参数的小模型，用字符级分词，只看 128 字上下文。ChatGPT 有上千亿参数、子词分词、数千字上下文窗口。在这个规模下，ppl≈31 已经是很好的结果了。

---

## 7. 下一步可以做什么？

当你跑通了整个流程，可以尝试以下方向来加深理解：

### 调参实验（零代码改动）
- 改 `epochs` 为 20 或 30 → 看更强的训练效果
- 改 `temperature` 为 0.5 或 1.5 → 看生成风格变化
- 改 `d_model` 为 128 或 512 → 感受模型容量对效果的影响

### 代码改动（加深理解）
- 换训练数据：《三国演义》《水浒传》《红楼梦》→ 看风格差异
- 把字符级分词换成 BPE 子词分词 → 理解现代分词器
- 加入学习率衰减（`lr_scheduler`）→ 提升训练效率
- 添加验证集，监控过拟合（训练 loss 降但验证 loss 升）

### 理论进阶
- 读 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- 读 [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy 的极简 GPT 实现
- 读 [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Karpathy 的 2 小时视频教程
- 参考本篇博客：[从零训练一个小 GPT](https://mcell.top/books/the-roads-for-llm/train-small-gpt)

---

## 8. 参考资料

- 本项目的架构和训练方法来自：[mcell.top — 从零训练一个小 GPT](https://mcell.top/books/the-roads-for-llm/train-small-gpt)
- 《西游记》原文来源：公开领域文本
- PyTorch 官方文档：[pytorch.org](https://pytorch.org)

---

## 项目文件

```
mini-gpt/
├── README.md           ← 你正在读的这个文件
├── train.py            ← 训练脚本（包含完整模型代码，~220 行）
├── generate.py         ← 交互式推理脚本
├── corpus.txt          ← 《西游记》UTF-8 纯文本（73 万字）
├── xiyouji_raw.txt     ← 原始文本（GB18030 编码）
└── minigpt.pt          ← 训练好的模型权重
```
