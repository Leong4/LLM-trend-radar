from typing import List

def count_tokens(s: str) -> int:
    # 轻量估算，后续可替换为 tiktoken
    return max(1, len((s or "").split()))

def window_chunks(text: str, max_tokens=400, overlap=60, min_tokens=60) -> List[str]:
    words = (text or "").split()
    n = len(words)
    out, i, idx = [], 0, 0
    if n == 0:
        return out
    while i < n:
        j = min(n, i + max_tokens)
        piece = " ".join(words[i:j])
        tokens = count_tokens(piece)
        if tokens >= min_tokens or (j == n and idx == 0):
            out.append(piece)
            idx += 1
        if j == n:
            break
        i = max(j - overlap, i + 1)
    return out