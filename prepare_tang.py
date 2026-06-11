"""
prepare_tang.py — 下载唐诗子集作为预训练语料，繁→简、清理。

用途: 两阶段训练的第一阶段（预训练）语料，让模型先学通用古诗语感。
数据源: https://github.com/chinese-poetry/chinese-poetry
产物: datas/corpus_tang.txt
"""

import os
import sys
import time
import requests

from clean import clean_text

try:
    import opencc
    _T2S = opencc.OpenCC("t2s")
except Exception:
    _T2S = None
    print("⚠ opencc 未装，繁体不转简体（建议 pip install opencc）")

REPO = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master"
TANG_FILES = [f"全唐诗/poet.tang.{i}.json" for i in range(0, 58000, 1000)]


def download_json(path):
    url = f"{REPO}/{path}"
    for _ in range(3):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return r.json()
        except Exception:
            time.sleep(1)
    return []


def main():
    # 默认累计到约 150 万字停止（命令行可覆盖，单位：字）
    target_chars = int(sys.argv[1]) if len(sys.argv) > 1 else 1_500_000

    os.makedirs("datas", exist_ok=True)
    out_path = "datas/corpus_tang.txt"

    total_chars = 0
    poems = 0

    print(f"下载唐诗，目标累计约 {target_chars:,} 字...")
    with open(out_path, "w", encoding="utf-8") as f:
        for path in TANG_FILES:
            if total_chars >= target_chars:
                break
            data = download_json(path)
            if not data:
                continue
            for item in data:
                title = item.get("title", "") or ""
                paras = item.get("paragraphs", [])
                if _T2S:
                    title = _T2S.convert(title)

                lines = []
                for p in paras:
                    if _T2S:
                        p = _T2S.convert(p)
                    cl = clean_text(p).strip()
                    if cl:
                        lines.append(cl)
                if not lines:
                    continue

                header = clean_text(title).strip()
                f.write(header + "\n")
                for line in lines:
                    f.write(line + "\n")
                f.write("\n")
                total_chars += sum(len(l) for l in lines)
                poems += 1
            print(f"  {path} → {poems:,} 首 / {total_chars:,} 字", end="\r")
    print()
    print(f"✅ {out_path}: {poems:,} 首, {total_chars:,} 字")


if __name__ == "__main__":
    main()
