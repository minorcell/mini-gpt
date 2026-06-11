"""
prepare_libai.py — 从御定全唐诗中提取李白全部诗作，繁→简，输出训练语料。

数据源: https://github.com/chinese-poetry/chinese-poetry
产物: datas/corpus_libai.txt
"""

import requests
import json
import os
import time

from clean import clean_text

try:
    import opencc
    _T2S = opencc.OpenCC("t2s")
except Exception:
    _T2S = None

REPO = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%BE%A1%E5%AE%9A%E5%85%A8%E5%94%90%E8%A9%A9/json"


def download_volume(vol):
    for _ in range(3):
        try:
            r = requests.get(f"{REPO}/{vol:03d}.json", timeout=15)
            if r.status_code == 200:
                return r.json()
        except Exception:
            time.sleep(1)
    return []


def clean_line(line):
    if not line:
        return ""
    line = line.strip()
    if _T2S:
        line = _T2S.convert(line)
    return clean_text(line)


def main():
    os.makedirs("datas", exist_ok=True)

    all_poems = {}
    volumes_with_libai = []

    print("扫描 900 卷中李白的诗...")
    for vol in range(1, 901):
        data = download_volume(vol)
        if not data:
            continue
        for item in data:
            if item.get("author", "") == "李白":
                if vol not in all_poems:
                    all_poems[vol] = []
                    volumes_with_libai.append(vol)
                all_poems[vol].append(item)

    print(f"在 {len(volumes_with_libai)} 卷中找到李白诗: {volumes_with_libai}\n")

    # 按卷号排序写入
    total_poems = 0
    total_chars = 0

    with open("datas/corpus_libai.txt", "w", encoding="utf-8") as f:
        for vol in sorted(volumes_with_libai):
            for item in all_poems[vol]:
                title = item.get("title", "") or ""
                if _T2S:
                    title = _T2S.convert(title)
                title = clean_text(title)
                paragraphs = item.get("paragraphs", [])

                lines = []
                for p in paragraphs:
                    cl = clean_line(p)
                    if cl:
                        lines.append(cl)
                if not lines:
                    continue

                f.write(f"{title}\n")
                for line in lines:
                    f.write(line + "\n")
                f.write("\n")

                total_poems += 1
                total_chars += sum(len(l) for l in lines)

    print(f"✅ datas/corpus_libai.txt: {total_poems:,} 首, {total_chars:,} 字")
    print("\n=== 前 5 首预览 ===")
    with open("datas/corpus_libai.txt", encoding="utf-8") as f:
        print(f.read(800))


if __name__ == "__main__":
    main()
