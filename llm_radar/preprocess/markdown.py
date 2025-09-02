import re

def first_paragraphs(md_text: str, max_bytes=2048, max_pars=2) -> str:
    text = (md_text or "").strip()
    b = text.encode("utf-8")
    if len(b) > max_bytes:
        text = b[:max_bytes].decode("utf-8", "ignore")
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return "\n\n".join(parts[:max_pars]) if parts else text