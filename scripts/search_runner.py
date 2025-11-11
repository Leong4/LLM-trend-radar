#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal search runner for your LLM Trend Radar DB.
Modes:
  - fts    : SQLite FTS5 only (bm25)
  - faiss  : FAISS HNSW only (dense)
  - hybrid : normalized blend of FTS and FAISS (alpha for FAISS weight)

Requirements:
  - llm_radar.config.DATA_DIR points to your data/ directory
  - DB: data/trend_radar.db  with tables: fts_chunk (rowid=chunk_id), doc_chunk, doc
  - FAISS index (for faiss/hybrid): data/vec/faiss.index + faiss.meta.json
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

from llm_radar.config import DATA_DIR
from llm_radar.retrieval.embedder import LocalSBERTEmbedder

DB_PATH = Path(DATA_DIR) / "trend_radar.db"
VEC_DIR = Path(DATA_DIR) / "vec"
INDEX_PATH = VEC_DIR / "faiss.index"
META_PATH  = VEC_DIR / "faiss.meta.json"


# ---------- utils ----------
def normalize_scores(d: Dict[int, float]) -> Dict[int, float]:
    """min-max normalize scores to [0,1]; empty -> {}, single -> 1.0"""
    if not d:
        return {}
    vs = np.array(list(d.values()), dtype=np.float64)
    lo, hi = float(vs.min()), float(vs.max())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in d}
    return {k: float((v - lo) / (hi - lo)) for k, v in d.items()}

# Robust normalize for FTS (handle tiny spreads); returns {cid:score}
def _robust_norm_with_jitter(pairs: list[tuple[int, float]], jitter: float = 5e-2) -> dict[int, float]:
    out: dict[int, float] = {}
    if not pairs:
        return out
    vals = [s for _, s in pairs]
    lo, hi = min(vals), max(vals)
    spread = hi - lo
    if spread <= 1e-9 or (spread / max(hi, 1e-9)) <= 1e-6:
        # spread is too tiny — assign a visible descending ladder so it won’t print all 1.0000
        for i, (cid, _) in enumerate(pairs):
            out[int(cid)] = max(0.0, 1.0 - jitter * i)
        return out
    rng = spread
    for cid, s in pairs:
        out[int(cid)] = (s - lo) / rng
    return out


def fetch_chunk_infos(con: sqlite3.Connection, chunk_ids: List[int],
                      lang: str | None, tiers: List[str]) -> Dict[int, Tuple[int, str, str, str]]:
    """
    Return info for given chunk ids:
      cid -> (doc_id, source_type, lang, text)
    Also filter by lang/tiers here.
    """
    if not chunk_ids:
        return {}
    qs = ",".join(["?"] * len(chunk_ids))
    sql = f"""
    SELECT c.id, c.doc_id, d.source_type, d.lang, c.text
    FROM doc_chunk c
    JOIN doc d ON d.id = c.doc_id
    WHERE c.id IN ({qs})
      AND c.tier IN ({",".join(["?"]*len(tiers))})
      {"" if not lang else "AND d.lang = ?"}
    """
    params = [*chunk_ids, *tiers]
    if lang:
        params.append(lang)

    out: Dict[int, Tuple[int, str, str, str]] = {}
    for cid, doc_id, source, clang, text in con.execute(sql, params).fetchall():
        out[int(cid)] = (int(doc_id), str(source), str(clang), str(text or ""))
    return out


# ---------- FTS ----------
def search_fts(con: sqlite3.Connection, query: str, k: int,
               lang: str | None, tiers: List[str]) -> Dict[int, float]:
    """
    Return {chunk_id: score} where score is larger=better.
    Uses bm25(fts_chunk) (lower is better), so convert to 1/(1+bm25).
    Also filters by lang/tiers via join to doc_chunk/doc.
    """
    tier_placeholders = ",".join(["?"] * len(tiers))
    sql = f"""
    SELECT c.id, bm25(fts_chunk) AS r
    FROM fts_chunk
    JOIN doc_chunk c ON c.id = fts_chunk.rowid
    JOIN doc d ON d.id = c.doc_id
    WHERE c.tier IN ({tier_placeholders})
      {"" if not lang else "AND d.lang = ?"}
      AND fts_chunk MATCH ?
    ORDER BY r ASC, length(c.text) DESC, c.id DESC
    LIMIT ?
    """
    params = [*tiers]
    if lang:
        params.append(lang)
    params.extend([query, int(k)])

    out: Dict[int, float] = {}
    try:
        rows = con.execute(sql, params).fetchall()
        pairs = [(int(cid), 1.0 / (1.0 + float(r))) for cid, r in rows]
        out = _robust_norm_with_jitter(pairs)
    except sqlite3.OperationalError as e:
        logger.error(f"FTS query failed: {e}")

    # fallback: if no hits, retry with a stricter AND-prefix query like "rag* AND faiss*"
    if not out:
        toks = [t for t in query.replace('"', ' ').replace("'", " ").split() if len(t) >= 2]
        if toks:
            # prefer AND to avoid huge OR slabs with identical bm25
            q2 = " AND ".join([f"{t}*" for t in toks[:4]])
            params2 = [*tiers]
            if lang:
                params2.append(lang)
            params2.extend([q2, int(k)])
            try:
                rows2 = con.execute(sql, params2).fetchall()
                pairs2 = [(int(cid), 1.0 / (1.0 + float(r))) for cid, r in rows2]
                if pairs2:
                    out = _robust_norm_with_jitter(pairs2)
            except sqlite3.OperationalError as e:
                logger.error(f"FTS fallback query failed: {e}")

    # second fallback: if still no hits, try a broader OR-prefix query like "rag* OR faiss* OR ..."
    if not out:
        toks = [t for t in query.replace('"', ' ').replace("'", " ").split() if len(t) >= 2]
        if toks:
            q3 = " OR ".join([f"{t}*" for t in toks[:6]])
            params3 = [*tiers]
            if lang:
                params3.append(lang)
            params3.extend([q3, int(k)])
            try:
                rows3 = con.execute(sql, params3).fetchall()
                pairs3 = [(int(cid), 1.0 / (1.0 + float(r))) for cid, r in rows3]
                if pairs3:
                    out = _robust_norm_with_jitter(pairs3)
            except sqlite3.OperationalError as e:
                logger.error(f"FTS OR-fallback query failed: {e}")
    return out


# ---------- FAISS ----------
def load_faiss_objects():
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("faiss is not available in this Python environment.") from e

    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise RuntimeError(f"FAISS files not found at {VEC_DIR}. Build it first with embed_runner.")
    index = faiss.read_index(str(INDEX_PATH))

    # detect model from meta
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    model_name = meta.get("model", "BAAI/bge-small-en-v1.5")
    # allow meta to store "sbert:..." style; strip the prefix for HF hub
    hf_name = model_name.split(":", 1)[1] if isinstance(model_name, str) and model_name.startswith("sbert:") else model_name
    embedder = LocalSBERTEmbedder(hf_name)
    return faiss, index, embedder


def search_faiss(query: str, k: int, lang: str | None, tiers: List[str]) -> Dict[int, float]:
    """
    Return {chunk_id: score} from FAISS (larger=better).
    Score uses 1/(1+L2) since we use HNSWFlat (L2).
    """
    faiss, index, embedder = load_faiss_objects()
    vec = embedder.encode([query]).astype("float32")
    D, I = index.search(vec, int(k))
    # I shape: (1,k); D shape: (1,k)
    out: Dict[int, float] = {}
    for cid, dist in zip(I[0].tolist(), D[0].tolist()):
        if cid == -1:
            continue
        out[int(cid)] = 1.0 / (1.0 + float(dist))
    return out


# ---------- main ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Search runner for FTS/FAISS/Hybrid")
    ap.add_argument("--q", required=True, help="query text")
    ap.add_argument("--mode", choices=["fts", "faiss", "hybrid"], default="fts")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5, help="FAISS weight for hybrid [0..1]")
    ap.add_argument("--lang", default=None, help="filter by doc.lang (optional)")
    ap.add_argument("--tiers", default="full,probation-lite", help="comma-separated tiers to consider")
    return ap.parse_args()


def pretty_print(con: sqlite3.Connection, ranking: List[Tuple[int, float]],
                 lang: str | None, tiers: List[str],
                 show_fields: bool = True):
    cids = [cid for cid, _ in ranking]
    info = fetch_chunk_infos(con, cids, lang, tiers)

    print("\n=== RESULTS ===")
    rnk = 1
    for cid, score in ranking:
        if cid not in info:
            continue
        doc_id, source, clang, text = info[cid]
        head = text.replace("\n", " ").strip()[:200]
        print(f"[{rnk:02d}] cid={cid} | doc={doc_id} | src={source} | lang={clang} | score={score:.4f}")
        print(f"     {head}")
        rnk += 1


def main():
    args = parse_args()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    K_collect = max(args.topk * 5, 50)  # collect more, filter later

    con = sqlite3.connect(str(DB_PATH))

    if args.mode == "fts":
        fts = search_fts(con, args.q, K_collect, args.lang, tiers)
        # keep order by score desc
        ranking = sorted(fts.items(), key=lambda x: x[1], reverse=True)[:args.topk]
        pretty_print(con, ranking, args.lang, tiers)
        return

    if args.mode == "faiss":
        fa = search_faiss(args.q, K_collect, args.lang, tiers)
        ranking = sorted(fa.items(), key=lambda x: x[1], reverse=True)[:args.topk]
        pretty_print(con, ranking, args.lang, tiers)
        return

    # hybrid
    fts = search_fts(con, args.q, K_collect, args.lang, tiers)
    fa = search_faiss(args.q, K_collect, args.lang, tiers)

    # normalize each channel separately, then blend
    fts_n = normalize_scores(fts)
    fa_n  = normalize_scores(fa)

    alpha = max(0.0, min(1.0, float(args.alpha)))
    all_ids = set(fts_n) | set(fa_n)
    fused: Dict[int, float] = {}
    for cid in all_ids:
        s_fts = fts_n.get(cid, 0.0)
        s_fa  = fa_n.get(cid, 0.0)
        fused[cid] = (1 - alpha) * s_fts + alpha * s_fa

    ranking = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:args.topk]
    pretty_print(con, ranking, args.lang, tiers)


if __name__ == "__main__":
    main()