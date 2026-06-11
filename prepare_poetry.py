"""
prepare_poetry.py — 下载唐诗宋词，繁→简，输出干净训练语料。

数据源: https://github.com/chinese-poetry/chinese-poetry
产物: datas/corpus_poetry.txt
"""

import os
import sys
import time
import requests

try:
    import opencc
    _T2S = opencc.OpenCC("t2s")
except Exception:
    _T2S = None
    print("⚠ opencc 未装，繁体不转简体（建议 pip install opencc）")

REPO = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master"

TANG_FILES = [f"全唐诗/poet.song.{i}.json" for i in range(0, 58000, 1000)]
SONG_CI_FILES = [f"宋词/ci.song.{i}.json" for i in range(0, 22000, 1000)]
EXTRAS = ["宋词/宋词三百首.json"]


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


def clean_line(line):
    if not line:
        return ""
    missing = line.count("□") + line.count("■")
    if missing > len(line) / 3:
        return ""
    line = line.replace("□", "").replace("■", "")
    if _T2S:
        line = _T2S.convert(line)
    return line.strip()


def main():
    max_tang = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    max_ci = int(sys.argv[2]) if len(sys.argv) > 2 else 24

    os.makedirs("datas", exist_ok=True)
    out_path = "datas/corpus_poetry.txt"

    total_chars = 0
    poems = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for label, files, limit in [
            ("唐诗", TANG_FILES, max_tang),
            ("宋词", SONG_CI_FILES, max_ci),
            ("精选", EXTRAS, len(EXTRAS)),
        ]:
            print(f"\n=== {label} ({min(limit, len(files))} 文件) ===")
            for path in files[:limit]:
                data = download_json(path)
                if not data:
                    continue
                for item in data:
                    paras = item.get("paragraphs", [])
                    title = item.get("title", "") or ""
                    author = item.get("author", "") or ""
                    rhythmic = item.get("rhythmic", "") or ""

                    if _T2S:
                        title = _T2S.convert(title)
                        author = _T2S.convert(author)
                        rhythmic = _T2S.convert(rhythmic)

                    lines = []
                    for p in paras:
                        cl = clean_line(p)
                        if cl:
                            lines.append(cl)
                    if not lines:
                        continue

                    header = title
                    if rhythmic:
                        header += f"（{rhythmic}）"
                    if author:
                        header += f" — {author}"

                    f.write(header + "\n")
                    for line in lines:
                        f.write(line + "\n")
                    f.write("\n")
                    total_chars += sum(len(l) for l in lines)
                    poems += 1
                print(f"  {path} → {poems:,} 首 / {total_chars:,} 字", end="\r")
            print()

    print(f"\n✅ {out_path}: {poems:,} 首, {total_chars:,} 字")
    print("=== 前 600 字预览 ===")
    with open(out_path, encoding="utf-8") as f:
        print(f.read(600))


if __name__ == "__main__":
    main()
