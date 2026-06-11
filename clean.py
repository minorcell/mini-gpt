"""
clean.py — 统一的字符清理，供 prepare_*.py 共用。

只保留：常用汉字（含扩展区 A）+ 中文标点 + 换行。
过滤：ASCII 字母数字、缺字标记 □■、生僻异体字（CJK 扩展 B 及以上的极罕见字）。
"""

# 训练语料中允许保留的中文标点
PUNCT = set("，。！？、：；「」『』（）《》〈〉…—")


def is_valid_char(c):
    """判断单个字符是否保留。"""
    if c == "\n":
        return True
    if c in PUNCT:
        return True
    # 基本汉字 + 扩展 A（覆盖唐诗宋词绝大多数用字）
    if "一" <= c <= "鿿":
        return True
    if "㐀" <= c <= "䶿":
        return True
    return False


def clean_text(text):
    """逐字过滤，去掉所有噪声字符。"""
    return "".join(c for c in text if is_valid_char(c))
