# 唐诗宋词 · MiniGPT

> 从零训练一个会写诗的小型 GPT 模型——用 7 万首唐诗宋词作教材，在普通笔记本上跑完全流程。


## 它是什么？

一个字符级 Decoder-only Transformer，用 **71,325 首唐诗宋词**（约 650 万字）训练而成。给它一个开头（如"春风""明月""长安"），它就能续写出古典诗句。

- 模型只有 **950 万** 个参数（ChatGPT 有上千亿）
- Apple Silicon MacBook 上训练约 **1.5 小时**
- 训练产物只有 **36 MB**，一行代码即可加载生成

它的定位就如编程入门时的 `printf("Hello World")`——麻雀虽小，五脏俱全。通过这个项目，你会亲手跑通 LLM 的完整流程：**数据处理 → 分词 → 模型搭建 → 训练 → 生成**。


## 快速开始

```bash
# 1. 安装依赖
pip install torch opencc

# 2. 拉取唐诗宋词数据（繁→简，清洗）
python3 prepare_poetry.py

# 3. 训练模型（CPU 稳定模式，约 1.5 小时）
python3 train.py --cpu

# 4. 交互式生成
python3 generate.py
```

训练完成后，Prompt 输入 `春风`、`明月`、`长安` 等开头，模型续写诗句。


## 项目结构

```
.
├── prepare_poetry.py     # 数据准备：下载唐诗宋词 → 繁转简 → 清洗
├── train.py              # 模型定义 + 训练循环（约 220 行）
├── generate.py           # 交互式生成脚本
├── datas/                # 清洗后的语料（corpus_poetry.txt）
│   └── corpus_poetry.txt
├── minigpt.pt            # 训练好的模型权重（36 MB）
└── README.md
```

核心代码都在 `train.py` 一个文件里。`generate.py` 只有 45 行。


## 它做了什么？

一张图概括：

```
chinese-poetry 仓库（JSON，繁体）
        │
        ▼  prepare_poetry.py
  繁→简（opencc）+ 去缺字 + 格式化
        │
        ▼  datas/corpus_poetry.txt（650 万字简体古诗）
        │
        ▼  CharTokenizer
  字符级分词器（9230 个不重复汉字 → 0~9229 的 ID）
        │
        ▼  get_batch()
  随机采样：输入前 128 字，预测下一个字
        │
        ▼  MiniGPT
  6 层 Transformer（9.5M 参数）
        │
        ▼  AdamW + 交叉熵损失
  训练 5 轮
        │
        ▼
  minigpt.pt → generate.py → 续写诗句
```


## 数据

数据来自 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) 仓库——目前最全的中华古典诗词开源数据集。

| 项目 | 数值 |
|------|------|
| 数据源 | 全唐诗 5 万首 + 宋词 2.1 万首 + 宋词三百首 |
| 原始格式 | JSON（繁体中文） |
| 清洗后 | **简体中文**，去缺字，去纯标题 |
| 总字符数 | **650 万字** |
| 不重复汉字 | **9,230 个** |


## 模型结构

MiniGPT 是一个 Decoder-only Transformer，由三部分拼成：

```
输入 ID（B, T）
    │
    ├── Token Embedding（字 → 256 维向量）
    ├── Position Embedding（位置 → 256 维向量）
    │
    ▼
相加 → 6 层 TransformerBlock → LayerNorm → Linear → 输出 logits
```

每层 TransformerBlock 做两件事：
1. **Causal Self-Attention**：预测第 N 个字时，回顾前 N-1 个字，找到最相关的信息
2. **Feed-Forward Network（FFN）**：把每个字的表示放大（256→1024），过 GELU 激活，再缩回

| 超参数 | 值 | 含义 |
|--------|-----|------|
| `d_model` | 256 | 词向量维度 |
| `n_heads` | 8 | 注意力头数 |
| `n_layers` | 6 | Transformer 层数 |
| `block_size` | 128 | 上下文窗口（一次最多看 128 字） |
| `batch_size` | 32 | 每次并行训练的样本数 |
| `vocab_size` | 9,230 | 词表大小 |


## 训练过程

### 损失曲线

| 轮次 | Loss | PPL | 说明 |
|------|------|-----|------|
| 1 | 5.00 | 148 | 瞎猜阶段——每字从 148 个候选里挑 |
| 2 | 4.25 | 70 | 学会了常见字搭配 |
| 3 | 3.99 | 54 | 五言、七言节奏浮现 |
| 4 | 3.85 | 47 | 开始有诗意 |
| 5 | **3.73** | **42** | 能写完整诗句 |

### 训练中的样本

- epoch 1: `太白仙人，锦袍初拥。`
- epoch 3: `像祭，广源无复纳降旗。`
- epoch 5: `飞花，花有数、愁无数。`

PPL（困惑度）可以理解为：**模型在每个位置平均犹豫几个选项**。PPL=42 意味着模型在每个字的位置从约 42 个候选里做选择——相比初始的 9,230 个选项，已经非常有把握了。


## 代码拆解

### 1. 字符级分词器

```python
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))  # 统计所有不重复字符
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # 字 → ID
        self.itos = {i: ch for i, ch in enumerate(chars)}  # ID → 字
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[ch] for ch in s if ch in self.stoi]

    def decode(self, ids):
        return ''.join([self.itos.get(i, '?') for i in ids])
```

为什么用字符级而不是 BPE 子词？**古典诗歌中每个汉字都是独立的语义单元**——"春风又绿江南岸"，每个字都不可拆。字符级对古诗既简单又精准。

### 2. 训练样本：教模型预测下一个字

```python
def get_batch(data, block_size, batch_size):
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y
```

`y` 是 `x` 整体右移一位：

```
x = ["春", "风", "又", "绿", "江"]
y = ["风", "又", "绿", "江", "南"]
```

这就是语言模型的标准训练范式——**根据上文预测下一个字**。不需要人工标注，文本本身就是标签。

### 3. Causal Self-Attention（因果自注意力）

```python
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
```

`causal_mask` 保证模型**不能偷看后面的字**——预测第 5 个字时，只能看到前 4 个。上三角矩阵把未来位置全部遮住。

### 4. 训练循环

```python
for epoch in range(epochs):
    for step in range(steps_per_epoch):
        x, y = get_batch(data, block_size, batch_size)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

4 步走：**采样 → 前向 → 反向 → 更新**。重复几万次。

### 5. 生成：自回归采样

```python
def generate(self, idx, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.block_size:]  # 截断到窗口大小
        logits = self(idx_cond)
        logits = logits[:, -1, :] / temperature  # 只看最后一步，调温度
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

每次只取模型输出的**最后一个位置**的概率分布，按 temperature 缩放后随机采样一个字，拼到序列末尾——如此循环。


## 关键概念速查

| 概念 | 一句话解释 |
|------|-----------|
| Token（词元） | 文字的数字编号。这里每个汉字就是一个 token |
| Embedding（嵌入） | 把 token ID 变成高维向量，捕捉字的语义 |
| Attention（注意力） | 让模型在预测时自动找到前文中相关的字 |
| Causal Mask | 保证只看过去，不看未来 |
| Loss（损失） | 模型预测错误的程度，越小越好 |
| PPL（困惑度） | `exp(loss)`，模型在每个位置犹豫几个选项 |
| Temperature | 生成的"创造力"——越高越狂野，越低越保守 |


## 自己跑

```bash
# 完整流程
python3 prepare_poetry.py        # ① 下载数据 → datas/corpus_poetry.txt
python3 train.py --cpu           # ② 训练 → minigpt.pt（约 1.5 小时）
python3 generate.py              # ③ 交互生成

# 快速验证（少量数据 + 1 轮）
python3 prepare_poetry.py 5 3    # 少拉点数据
python3 train.py --cpu           # 把 epochs 改成 1
```

`generate.py` 参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--ckpt` | `minigpt.pt` | 模型文件 |
| `--temperature` | `0.8` | 创造性（0.5=保守 1.2=狂野） |
| `--max-new-tokens` | `200` | 最多生成多少字 |


## 参考

- [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) — 中华古典诗词数据集
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- [karpathy/minGPT](https://github.com/karpathy/minGPT) — 本项目架构的灵感来源
