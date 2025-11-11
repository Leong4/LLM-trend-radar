#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal RAG QA runner for LLM Trend Radar.
- Reuses search functions from scripts.search_runner
- Builds context windows, then calls an LLM (ollama or OpenAI-compatible)
"""

import os
import json
import argparse
import sqlite3
import textwrap
from pathlib import Path
from typing import List, Tuple, Dict

from loguru import logger

# 复用你已有的检索与取文函数


from scripts.search_runner import (
    DB_PATH,  # data/trend_radar.db
    search_fts, search_faiss, normalize_scores, fetch_chunk_infos
)

# -------- Intent detection & prompt templates --------
def detect_intent(q: str) -> str:
    qs = q.lower()
    if " vs " in qs or "versus" in qs or "compare" in qs or "comparison" in qs:
        return "compare"
    if "how to" in qs or "steps" in qs or "pipeline" in qs or "implementation" in qs:
        return "howto"
    if "evaluate" in qs or "metric" in qs or "benchmark" in qs:
        return "evaluate"
    if "trend" in qs or "timeline" in qs or "roadmap" in qs:
        return "trend"
    return "qa"

PROMPTS = {
    "qa": """You are an expert on LLM/RAG trends. Answer strictly from CONTEXT.\nIf missing, say you don't know. Cite like [S1],[S2]. Be concise.\n\n# QUESTION\n{query}\n\n# CONTEXT\n{context}\n\n# OUTPUT\n- 3–6 bullet points\n- Inline citations [S#]\n""",
    "compare": """Compare the items mentioned in QUESTION using only CONTEXT.\nIf any claim lacks support, omit it. Cite [S#].\n\n# QUESTION\n{query}\n\n# CONTEXT\n{context}\n\n# OUTPUT (Markdown table + bullets)\n- A short pros/cons table\n- 3–5 bullets of takeaways with citations\n""",
    "howto": """Provide a step-by-step procedure based only on CONTEXT.\nIf steps are incomplete in CONTEXT, say what's missing. Cite [S#].\n\n# QUESTION\n{query}\n\n# CONTEXT\n{context}\n\n# OUTPUT\n- Numbered steps\n- Notes/risks with citations\n""",
    "evaluate": """Describe evaluation methods and metrics strictly from CONTEXT.\nIf metric values are not in CONTEXT, don't fabricate. Cite [S#].\n\n# QUESTION\n{query}\n\n# CONTEXT\n{context}\n\n# OUTPUT\n- Metrics list with brief definitions\n- Datasets/benchmarks (if present) with [S#]\n""",
    "trend": """Summarize trends/roadmap strictly from CONTEXT. Avoid speculation. Cite [S#].\n\n# QUESTION\n{query}\n\n# CONTEXT\n{context}\n\n# OUTPUT\n- 3–6 trend bullets with timeframe if available\n- Any notable releases with [S#]\n"""
}

# --- FTS safety helpers ---
def _sanitize_fts_query(q: str) -> str | None:
    """Sanitize user text into a *safe* FTS5 MATCH query.
    Key decisions:
    - remove punctuation and wildcards that break MATCH (e.g. * ? " ( ) [ ] { } : )
    - drop boolean keywords (and/or/not/near) so the tokenizer won't leave a naked operator
    - join terms with whitespace (implicit AND) instead of explicit AND to survive stopword removal
    - cap to first 8 tokens; return None if nothing usable
    """
    import re
    # keep word characters in unicode; everything else becomes a space
    terms = re.findall(r"\w+", q, flags=re.UNICODE)
    if not terms:
        return None
    reserved = {"and", "or", "not", "near", "NEAR"}
    # drop very short tokens (length==1) and reserved words
    cleaned = [t for t in terms if len(t) > 1 and t.lower() not in reserved]
    if not cleaned:
        return None
    # Use whitespace-separated tokens (implicit AND in FTS5). No quotes, no explicit AND.
    return " ".join(cleaned[:8])


def _safe_search_fts(con, q: str | None, k: int, lang, tiers):
    """Wrap search_fts to prevent hard failures; return empty dict on error."""
    if not q:
        return {}
    try:
        return search_fts(con, q, k, lang, tiers)
    except Exception as e:
        logger.warning("FTS failed: {} — falling back to FAISS-only.", e)
        return {}

# =========================
# LLM backends
# =========================

def call_ollama(model: str, prompt: str, host: str = "http://localhost:11434") -> str:
    import requests
    url = f"{host}/api/generate"
    data = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=data, timeout=300)
    r.raise_for_status()
    obj = r.json()
    return obj.get("response", "").strip()

def call_openai_compat(model: str, prompt: str, base_url: str, api_key: str) -> str:
    import requests
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful research assistant. Answer ONLY with facts from the provided context."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    r = requests.post(url, headers=headers, json=data, timeout=300)
    r.raise_for_status()
    obj = r.json()
    return obj["choices"][0]["message"]["content"].strip()


# =========================
# Utils: 拼窗 / 限长 / 去重
# =========================

# --- must filter & snippets ---
def apply_must_filter(con: sqlite3.Connection,
                      ranking: List[Tuple[int, float]],
                      lang, tiers, must_terms: list[str]) -> List[Tuple[int, float]]:
    """Filter ranking by must terms (case-insensitive). If all filtered out, fall back to original."""
    if not must_terms:
        return ranking
    cids = [cid for cid, _ in ranking]
    info = fetch_chunk_infos(con, cids, lang, tiers)  # cid -> (doc_id, source, lang, text)
    must_lc = [t.strip().lower() for t in must_terms if t.strip()]
    kept: List[Tuple[int, float]] = []
    # fetch titles once
    doc_ids = list({info[cid][0] for cid in cids if cid in info})
    title_map = {}
    if doc_ids:
        q = "SELECT id, title FROM doc WHERE id IN ({})".format(
            ",".join("?" * len(doc_ids))
        )
        for did, title in con.execute(q, doc_ids).fetchall():
            title_map[did] = (title or "").lower()
    for cid, score in ranking:
        if cid not in info:
            continue
        did, _, _, text = info[cid]
        t = (text or "").lower()
        title_lc = title_map.get(did, "")
        if any(mt in t or mt in title_lc for mt in must_lc):
            kept.append((cid, score))
    return kept if kept else ranking


def make_snippets(con: sqlite3.Connection, cids: List[int], max_len: int = 200) -> Dict[int, str]:
    if not cids:
        return {}
    q = "SELECT id, text FROM doc_chunk WHERE id IN ({})".format(
        ",".join("?" * len(cids))
    )
    out: Dict[int, str] = {}
    for cid, tx in con.execute(q, cids).fetchall():
        s = (tx or "").strip().replace("\n", " ")
        out[int(cid)] = s[:max_len]
    return out

def fetch_doc_meta(con: sqlite3.Connection, doc_ids):
    """Fetch minimal metadata for display (title, source_type)."""
    if not doc_ids:
        return {}
    q = "SELECT id, title, source_type FROM doc WHERE id IN ({})".format(
        ",".join("?" * len(doc_ids))
    )
    meta = {}
    for did, title, src in con.execute(q, list(doc_ids)).fetchall():
        meta[did] = {"title": title or f"Doc {did}", "source": src}
    return meta



def format_sources_with_titles(sources, meta):
    """Format sources block with titles for prompts / printing."""
    lines = []
    for tag, did in sources:
        m = meta.get(did, {})
        title = m.get("title", f"Doc {did}")
        src   = m.get("source", "?")
        lines.append(f"- [{tag}] doc_id={did} | {src} | {title}")
    return "\n".join(lines) if lines else "（无检索来源）"

# --- Simple MMR (token Jaccard) to diversify candidate chunks ---
def _token_set(s: str) -> set[str]:
    import re
    return set(t.lower() for t in re.findall(r"\w+", s)) - {
        "the","and","of","to","a","in","for","on","with","is","are","as","at","by"
    }

def mmr_select_blocks(blocks: List[Tuple[int, str]], k: int = 12, lam: float = 0.7) -> List[Tuple[int, str]]:
    """
    blocks: [(doc_id, text)]
    k: number of blocks to keep across docs
    lam: balance between relevance (keep original order as implicit relevance) and diversity (Jaccard)
    """
    if not blocks:
        return []

    # implicit relevance: earlier blocks are more relevant
    rel = {i: (len(blocks) - idx) / len(blocks) for idx, i in enumerate(range(len(blocks)))}

    # precompute token sets
    toks = [_token_set(t) for _, t in blocks]

    def max_sim_to_selected(j: int, selected_idx: List[int]) -> float:
        if not selected_idx:
            return 0.0
        tj = toks[j]
        sims = []
        for i in selected_idx:
            ti = toks[i]
            inter = len(ti & tj)
            union = len(ti | tj) or 1
            sims.append(inter / union)
        return max(sims) if sims else 0.0

    selected: List[int] = []
    candidates = list(range(len(blocks)))

    while candidates and len(selected) < k:
        best_i, best_score = None, -1.0
        for j in candidates:
            score = lam * rel[j] - (1 - lam) * max_sim_to_selected(j, selected)
            if score > best_score:
                best_i, best_score = j, score
        selected.append(best_i)
        candidates.remove(best_i)

    return [blocks[i] for i in sorted(selected)]

def approx_tokens(s: str) -> int:
    # 简单估计：英文 1 token ≈ 4 chars；中文 1 token ≈ 2 chars；这里用一个保守折中
    return max(1, len(s) // 3)

def build_context(con: sqlite3.Connection,
                  ranking: List[Tuple[int, float]],
                  lang: str | None,
                  tiers: List[str],
                  max_ctx_tokens: int = 1500,
                  max_per_doc: int = 2) -> Tuple[str, List[Tuple[str, int]]]:
    """
    根据检索结果取文本，按 doc_id 分组拼窗，限长裁剪。
    返回:
      context_text: 带 [S1]/[S2] 标记的拼接文本
      sources: [(tag, doc_id)] 例如 [("S1", 49), ("S2", 24)]
    """
    cids = [cid for cid, _ in ranking]
    info = fetch_chunk_infos(con, cids, lang, tiers)  # cid -> (doc_id, source, lang, text)

    # 先按 doc_id 分桶
    by_doc: Dict[int, List[Tuple[int, str]]] = {}
    for cid, (_, _, _, text) in info.items():
        doc_id = info[cid][0]
        by_doc.setdefault(doc_id, []).append((cid, text))

    # 每个 doc 取前 max_per_doc 个块，并简单拼接
    blocks: List[Tuple[int, str]] = []
    for doc_id, items in by_doc.items():
        # 保持 ranking 的顺序优先
        items_sorted = sorted(items, key=lambda x: cids.index(x[0]))
        merged = []
        buf = []
        cur_tokens = 0
        for cid, t in items_sorted:
            if t is None:
                continue
            if approx_tokens(t) > 400:  # 长块截断，避免爆长度
                t = t[:1500]
            buf.append(t)
            cur_tokens += approx_tokens(t)
            if cur_tokens >= 400:  # 一个小窗 ≈ 400 tokens
                merged.append(" ".join(buf))
                buf, cur_tokens = [], 0
        if buf:
            merged.append(" ".join(buf))
        # 取前 N 个小窗
        for piece in merged[:max_per_doc]:
            blocks.append((doc_id, piece))

    # across-doc MMR：在全体小窗层面做一次去冗余，保留更丰富的证据面
    blocks = mmr_select_blocks(blocks, k=min(12, len(blocks)), lam=0.7)

    # 重新按检索分数顺序组织 doc 顺序
    doc_order = []
    for cid, _ in ranking:
        did = info.get(cid, (None, None, None, None))[0]
        if did and did not in doc_order:
            doc_order.append(did)

    # 组装上下文，限总长
    ctx_parts = []
    sources: List[Tuple[str, int]] = []
    tag_id = 1
    used_tokens = 0
    for did in doc_order:
        for d, piece in blocks:
            if d != did:
                continue
            tk = approx_tokens(piece)
            if used_tokens + tk > max_ctx_tokens:
                continue
            tag = f"S{tag_id}"
            ctx_parts.append(f"[{tag}] {piece}")
            sources.append((tag, did))
            tag_id += 1
            used_tokens += tk

    return "\n\n".join(ctx_parts), sources

def build_prompt(query: str, context: str, src_block: str) -> str:
    tpl = f"""
You are an expert on LLM/RAG trends. Answer the user strictly based on the CONTEXT.
If the answer is not in the context, say you don't know. Cite sources inline like [S1], [S2].

# QUESTION
{query}

# CONTEXT
{context}

# REQUIREMENTS
- Use concise sentences and bullet points when appropriate.
- Include inline citations [S#] tied to the source list below.
- If numbers or dates appear, quote them exactly from the context.

# SOURCES
{src_block}

# ANSWER
"""
    return textwrap.dedent(tpl).strip()

# =========================
# main
# =========================

def parse_args():
    ap = argparse.ArgumentParser(description="RAG QA runner")
    ap.add_argument("--q", required=True, help="question")
    ap.add_argument("--mode", choices=["fts", "faiss", "hybrid"], default="hybrid")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of FAISS in hybrid")
    ap.add_argument("--topk", type=int, default=8, help="collect K for retrieval (pre-merge)")
    ap.add_argument("--lang", default=None, help="filter by doc.lang")
    ap.add_argument("--tiers", default="full,probation-lite", help="comma separated tiers")
    ap.add_argument("--max_ctx_tokens", type=int, default=1500)
    ap.add_argument("--max_per_doc", type=int, default=2)

    # LLM settings
    ap.add_argument("--provider", choices=["ollama", "openai"], default=os.getenv("LLM_PROVIDER", "ollama"))
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "qwen2.5:14b"))
    ap.add_argument("--ollama_host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--openai_base", default=os.getenv("OPENAI_BASE", "http://localhost:8000"))
    ap.add_argument("--openai_key", default=os.getenv("OPENAI_API_KEY", "sk-xxx"))
    ap.add_argument("--dry_run", action="store_true", help="print context & exit")

    ap.add_argument("--emit", choices=["text", "json", "md"], default="text",
                    help="output format: plain text (default), JSON, or Markdown")
    ap.add_argument("--min_evidence", type=int, default=2,
                    help="min distinct docs required to answer; fallback if not met")
    ap.add_argument("--must", default=None,
                    help="comma-separated keywords that must appear in title or chunk text")
    ap.add_argument("--answer_style", choices=["brief", "detailed"], default="brief")
    ap.add_argument("--min_citations", type=int, default=1,
                    help="min number of distinct [S#] citations required in model answer before accepting")
    ap.add_argument("--retry_on_low_cite", action="store_true",
                    help="if set, will ask the model to regenerate once with explicit cite reminder")
    return ap.parse_args()

def main():
    args = parse_args()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]

    con = sqlite3.connect(str(DB_PATH))

    # 检索
    K_collect = max(args.topk * 5, 50)
    if args.mode == "fts":
        q_fts = _sanitize_fts_query(args.q)
        fts = _safe_search_fts(con, q_fts, K_collect, args.lang, tiers)
        ranking = sorted(fts.items(), key=lambda x: x[1], reverse=True)[:args.topk]
    elif args.mode == "faiss":
        fa = search_faiss(args.q, K_collect, args.lang, tiers)
        ranking = sorted(fa.items(), key=lambda x: x[1], reverse=True)[:args.topk]
    else:
        q_fts = _sanitize_fts_query(args.q)
        fts = _safe_search_fts(con, q_fts, K_collect, args.lang, tiers)
        fa  = search_faiss(args.q, K_collect, args.lang, tiers)
        fts_n = normalize_scores(fts); fa_n = normalize_scores(fa)
        alpha = max(0.0, min(1.0, float(args.alpha)))
        if not fts_n:  # FTS unavailable/failed → rely on FAISS only
            ranking = sorted(fa.items(), key=lambda x: x[1], reverse=True)[:args.topk]
        else:
            all_ids = set(fts_n) | set(fa_n)
            fused = {cid: (1-alpha)*fts_n.get(cid, 0.0) + alpha*fa_n.get(cid, 0.0) for cid in all_ids}
            ranking = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:args.topk]

    # must filter（有命中则生效，否则回退原集合）
    must_terms = [t.strip() for t in (args.must.split(",") if args.must else []) if t.strip()]
    ranking = apply_must_filter(con, ranking, args.lang, tiers, must_terms)

    # 取文 + 拼窗
    context, sources = build_context(
        con, ranking, args.lang, tiers,
        max_ctx_tokens=args.max_ctx_tokens, max_per_doc=args.max_per_doc
    )

    # Prepare source block with titles
    distinct_docs = list({did for _, did in sources})
    meta = fetch_doc_meta(con, distinct_docs)
    src_block = format_sources_with_titles(sources, meta)

    # If not enough evidence or empty context, short-circuit with a safe response
    if len(distinct_docs) < args.min_evidence or not context.strip():
        payload = {
            "answer": "I don't have enough evidence in the retrieved context to answer this question.",
            "sources": [{"tag": tag, "doc_id": did, **meta.get(did, {})} for tag, did in sources],
            "mode": args.mode, "topk": args.topk, "alpha": getattr(args, "alpha", None)
        }
        if args.emit == "json":
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        elif args.emit == "md":
            print("### Answer\n\n" + payload["answer"] + "\n\n### Sources\n" + src_block)
        else:
            print("\n=== ANSWER ===\n")
            print(payload["answer"])
            print("\n=== SOURCES ===\n" + src_block)
        return

    if args.dry_run:
        print("=== CONTEXT ===\n", context)
        print("\n=== SOURCES ===\n" + src_block)
        return

    intent = detect_intent(args.q)
    prompt_tmpl = PROMPTS.get(intent, PROMPTS["qa"])
    if args.answer_style == "detailed":
        prompt_tmpl += "\n- Provide a little more detail when helpful (but still grounded)."
    prompt = textwrap.dedent(prompt_tmpl.format(query=args.q, context=context)).strip()
    prompt += "\n\n# SOURCES\n" + src_block + "\n\n# ANSWER\n"

    # 调 LLM
    if args.provider == "ollama":
        answer = call_ollama(args.model, prompt, host=args.ollama_host)
    else:
        answer = call_openai_compat(args.model, prompt, base_url=args.openai_base, api_key=args.openai_key)

    # collect snippets
    cid_list = [cid for cid, _ in ranking]
    snippets = make_snippets(con, cid_list, max_len=200)

    # citation stats and guard
    import re as _re
    cites = _re.findall(r"\[S(\d+)\]", answer)
    citations_count = len(set(cites))

    # If citations are too few, optionally regenerate once with a reminder
    if citations_count < args.min_citations and args.retry_on_low_cite:
        regen_prompt = prompt + "\n\nReminder: Your answer must include at least one inline citation [S#] that maps to the source list."
        if args.provider == "ollama":
            answer = call_ollama(args.model, regen_prompt, host=args.ollama_host)
        else:
            answer = call_openai_compat(args.model, regen_prompt, base_url=args.openai_base, api_key=args.openai_key)
        cites = _re.findall(r"\[S(\d+)\]", answer)
        citations_count = len(set(cites))

    # If still insufficient, safe fallback
    if citations_count < args.min_citations:
        fallback = {
            "answer": "I don't have enough grounded evidence in the retrieved context to provide a cited answer.",
            "sources": [{"tag": tag, "doc_id": did, **meta.get(did, {})} for tag, did in sources],
            "mode": args.mode, "topk": args.topk, "alpha": getattr(args, "alpha", None),
            "intent": intent,
            "citations_count": citations_count,
            "evidence_docs": len(distinct_docs),
        }
        if args.emit == "json":
            print(json.dumps(fallback, ensure_ascii=False, indent=2))
        elif args.emit == "md":
            print("### Answer\n\n" + fallback["answer"] + "\n\n### Sources\n" + src_block)
        else:
            print("\n=== ANSWER ===\n" + fallback["answer"])
            print("\n=== SOURCES ===\n" + src_block)
        return

    # Normal payload
    payload = {
        "answer": answer,
        "sources": [{"tag": tag, "doc_id": did, **meta.get(did, {})} for tag, did in sources],
        "mode": args.mode, "topk": args.topk, "alpha": getattr(args, "alpha", None),
        "intent": intent,
        "citations_count": citations_count,
        "evidence_docs": len(distinct_docs),
        "snippets": {str(cid): snippets.get(cid, "") for cid in cid_list},
    }

    if args.emit == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.emit == "md":
        print("### Answer\n\n" + answer + "\n\n### Sources\n" + src_block)
    else:
        print("\n=== ANSWER ===\n" + answer)
        print("\n=== SOURCES ===\n" + src_block)

if __name__ == "__main__":
    main()