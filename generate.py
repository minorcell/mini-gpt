"""
交互式生成脚本 — 加载训练好的 MiniGPT 模型，输入 prompt 生成 西游记 风格文字。
"""

import torch
from train import MiniGPT, CharTokenizer, load_data


def main():
    # 加载语料构建分词器
    text = load_data("corpus.txt")
    tokenizer = CharTokenizer(text)

    # 创建模型并加载权重
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = MiniGPT(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load("minigpt.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"MiniGPT 已加载 | 参数: {n_params/1e6:.1f}M | 设备: {device}")
    print(f"词汇表: {tokenizer.vocab_size} | 上下文长度: {model.block_size}")
    print("输入 prompt 开始生成（输入 'quit' 退出）\n")

    while True:
        prompt = input("Prompt > ").strip()
        if prompt.lower() == 'quit':
            break
        if not prompt:
            prompt = "行者"

        # 编码 prompt
        ids = tokenizer.encode(prompt)
        if len(ids) > model.block_size:
            ids = ids[-model.block_size:]
        context = torch.tensor([ids], dtype=torch.long).to(device)

        # 生成
        gen = model.generate(context, max_new_tokens=200, temperature=0.8)
        output = tokenizer.decode(gen[0].tolist())
        print(f"\n{output}\n")


if __name__ == "__main__":
    main()
