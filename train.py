"""
MiniGPT — 基于李白诗歌训练一个字符级小 GPT。

架构和方法来自: https://mcell.top/books/the-roads-for-llm/train-small-gpt
"""

import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 分词器 ───────────────────────────────────────────────

class CharTokenizer:
    """字符级分词器：每个字符映射为一个整数 ID。

    两阶段训练需要预训练与微调共享同一词表，故支持从多语料并集
    构建（from_corpora）并持久化到磁盘（save/load）。
    """

    def __init__(self, chars):
        # chars 可以是一段文本，也可以是已排序的字符列表
        if isinstance(chars, str):
            chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # string → int
        self.itos = {i: ch for i, ch in enumerate(chars)}  # int → string
        self.vocab_size = len(chars)

    @classmethod
    def from_corpora(cls, filepaths):
        """从多个语料文件的字符并集构建词表（用于共享词表）。"""
        chars = set()
        for fp in filepaths:
            chars |= set(load_data(fp))
        return cls(sorted(chars))

    def save(self, path):
        chars = [self.itos[i] for i in range(self.vocab_size)]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(chars, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            chars = json.load(f)
        return cls(chars)

    def encode(self, s):
        return [self.stoi[ch] for ch in s if ch in self.stoi]

    def decode(self, ids):
        return ''.join([self.itos.get(i, '?') for i in ids])


# ── 数据加载 ─────────────────────────────────────────────

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text


# ── 批采样 ───────────────────────────────────────────────

def get_batch(data, block_size, batch_size):
    """随机采样 batch：输入是前 block_size 个 token，标签后移一位。"""
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# ── Transformer 块 ───────────────────────────────────────

class TransformerBlock(nn.Module):
    """Decoder-only 块：自注意力 + FFN，均带残差连接。"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Causal mask：token 只能看到自身及之前的 token
        causal_mask = torch.triu(
            torch.ones(x.size(1), x.size(1), device=x.device),
            diagonal=1
        ).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x


# ── MiniGPT 模型 ─────────────────────────────────────────

class MiniGPT(nn.Module):
    """字符级 Decoder-only Transformer，约 1000 万参数。"""

    def __init__(self, vocab_size, d_model=256, n_heads=8,
                 n_layers=6, block_size=128, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.block_size = block_size
        self.apply(self._init_weights)
        # 权重绑定：输入 embedding 和输出 head 共享权重（GPT-2 标配）
        # 减参数、防过拟合，小模型效果尤其明显
        self.lm_head.weight = self.token_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 因为权重绑定，embedding 初始化稍小以补偿
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)                    # (B, T, d_model)
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.pos_emb(pos)                      # (T, d_model)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                         # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成。top_k 只在概率最高的 k 个字中采样，过滤长尾乱码。"""
        for _ in range(max_new_tokens):
            # 截断到 block_size
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ── 训练循环 ─────────────────────────────────────────────

def get_lr(it, warmup, total, max_lr):
    """余弦学习率：warmup 线性上升 → 余弦衰减到 max_lr/10。"""
    if it < warmup:
        return max_lr * (it + 1) / warmup
    if it >= total:
        return max_lr / 10
    ratio = (it - warmup) / max(1, total - warmup)
    return max_lr / 10 + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - max_lr / 10)


def train(model, data, tokenizer, epochs=100, batch_size=32,
          block_size=128, lr=3e-4, device='cpu', weight_decay=0.1):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.95), weight_decay=weight_decay)
    steps_per_epoch = len(data) // (batch_size * block_size)
    total_steps = epochs * steps_per_epoch
    warmup = total_steps // 10  # 前 10% 步数做 warmup
    step_count = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step in range(steps_per_epoch):
            # 调整学习率
            lr_now = get_lr(step_count, warmup, total_steps, lr)
            for g in optimizer.param_groups:
                g['lr'] = lr_now

            x, y = get_batch(data, block_size, batch_size)
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step_count += 1

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"  step {step:>4d}/{steps_per_epoch}  loss = {loss.item():.4f}  lr = {lr_now:.1e}")

        avg_loss = total_loss / steps_per_epoch
        perplexity = math.exp(avg_loss)
        print(f"epoch {epoch + 1:>2d}/{epochs}  "
              f"loss = {avg_loss:.4f}  ppl = {perplexity:.1f}")

        # 每 5 轮生成一次样本
        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            # 随机取一段上下文作为 prompt
            start = torch.randint(0, len(data) - 10, (1,))
            context = data[start:start + 10].unsqueeze(0).to(device)
            gen = model.generate(context, max_new_tokens=80, temperature=0.8, top_k=40)
            sample = tokenizer.decode(gen[0].tolist())
            print(f"  [生成样本] {sample}")
            print()


# ── 共享词表 ─────────────────────────────────────────────

VOCAB_PATH = "datas/vocab.json"
BLOCK_SIZE = 128  # 诗短，128 字足够覆盖整首；训练与推理共用此值


def get_shared_tokenizer():
    """加载共享词表；不存在则从 唐诗+李白 并集构建并持久化。

    两阶段训练的预训练与微调必须共用同一词表，否则 embedding 对不齐。
    """
    if os.path.exists(VOCAB_PATH):
        return CharTokenizer.load(VOCAB_PATH)
    corpora = [p for p in ["datas/corpus_tang.txt", "datas/corpus_libai.txt"]
               if os.path.exists(p)]
    tok = CharTokenizer.from_corpora(corpora)
    tok.save(VOCAB_PATH)
    print(f"已构建共享词表 → {VOCAB_PATH}（{tok.vocab_size} 字，来自 {len(corpora)} 个语料）")
    return tok


def pick_device(force_cpu):
    if force_cpu:
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# ── 主流程 ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MiniGPT 训练（支持两阶段：预训练 / 微调）")
    parser.add_argument("--data", default="datas/corpus_libai.txt", help="训练语料路径")
    parser.add_argument("--out", default="minigpt.pt", help="模型输出路径")
    parser.add_argument("--init-from", default=None, help="微调起点：加载已有权重再训练")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=None, help="不指定时：从头训用 3e-4，微调用 1e-4")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # 1. 共享词表（并集，保证两阶段一致）
    tokenizer = get_shared_tokenizer()
    print(f"词汇表大小: {tokenizer.vocab_size}（共享词表）")

    # 2. 加载训练语料
    text = load_data(args.data)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"训练语料: {args.data}  {len(text):,} 字符 → {len(data):,} token")

    # 3. 创建模型
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        block_size=BLOCK_SIZE,
    )
    print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # 4. 微调：加载预训练权重
    if args.init_from:
        state = torch.load(args.init_from, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        print(f"已加载预训练权重: {args.init_from}（进入微调）")

    # 5. 学习率：微调默认更小，避免冲掉预训练知识
    lr = args.lr if args.lr is not None else (1e-4 if args.init_from else 3e-4)

    device = pick_device(args.cpu)
    print(f"训练设备: {device} | epochs: {args.epochs} | lr: {lr:.1e}")
    print(f"每轮步数: {len(data) // (32 * BLOCK_SIZE)}")
    print()

    # 6. 训练
    train(model, data, tokenizer, epochs=args.epochs,
          block_size=BLOCK_SIZE, lr=lr, device=device)

    # 7. 保存
    torch.save(model.state_dict(), args.out)
    print(f"模型已保存到 {args.out}")
