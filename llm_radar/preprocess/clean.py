import re, unicodedata

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t\u00A0]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def strip_math(s: str) -> str:
    s = re.sub(r"\$[^$\n]{1,200}\$", "[EQ]", s)
    s = re.sub(r"\\\[.*?\\\]", "[EQ]", s, flags=re.S)
    return s

def strip_code_blocks_md(s: str) -> str:
    return re.sub(r"```.*?```", "[CODE]", s, flags=re.S)