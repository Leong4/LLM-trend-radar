#!/usr/bin/env python3
import argparse, json, os, sqlite3, time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from loguru import logger

# ---- project imports ----
from llm_radar.config import DATA_DIR
from llm_radar.db.models import SessionLocal, DocORM, DocChunkORM
from llm_radar.retrieval.embedder import LocalSBERTEmbedder, TFIDFEmbedder, Embedder

DB_PATH = Path(DATA_DIR) / "trend_radar.db"
VEC_DIR = Path(DATA_DIR) / "vec"
VEC_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = VEC_DIR / "faiss.index"
META_PATH  = VEC_DIR / "faiss.meta.json"

try:
    import faiss  # type: ignore
    FAISS_OK = True
except Exception:
    FAISS_OK = False
    logger.warning("faiss-cpu not available; will only write vec_chunk + fts_chunk, skip FAISS index.")


def ensure_fts_chunk(rebuild: bool = False):
    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()
    cur.execute("BEGIN")
    try:
        if rebuild:
            
            cur.execute("DROP TABLE IF EXISTS fts_chunk;")
            # 按你库里的建表方式来；若之前是 contentless 就保留
            cur.execute("CREATE VIRTUAL TABLE fts_chunk USING fts5(text);")
        else:
            cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='fts_chunk';")
            if not cur.fetchone():
                cur.execute("CREATE VIRTUAL TABLE fts_chunk USING fts5(text);")
        con.commit()
    finally:
        con.close()
# ------------------ args ------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Embed chunks and build/append FAISS + FTS5")
    ap.add_argument("--model", choices=["sbert", "tfidf"], default="sbert")
    ap.add_argument("--sbert-name", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=10000, help="max chunks to embed this run")
    ap.add_argument("--rebuild", action="store_true", help="rebuild index from scratch")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--tiers", default="full,probation-lite")
    # HNSW params
    ap.add_argument("--hnsw-M", type=int, default=32)
    ap.add_argument("--hnsw-efC", type=int, default=200)
    ap.add_argument("--hnsw-efS", type=int, default=128)
    return ap.parse_args()

# ------------------ embedder factory ------------------
def make_embedder(args) -> Embedder:
    if args.model == "sbert":
        return LocalSBERTEmbedder(args.sbert_name)
    else:
        return TFIDFEmbedder(max_features=4096)

# ------------------ db helpers ------------------
def select_candidate_chunks(limit:int, lang:str, tiers:List[str], model_name:str, rebuild:bool) -> List[Tuple[int,str]]:
    """
    返回待嵌入的 (chunk_id, text)
    - 仅选语言&层级符合的 chunk
    - 若非 rebuild，只挑选 vec_chunk 里还不存在指定 model 的
    """
    tiers_tuple = tuple(tiers)
    with SessionLocal() as s:
        # 连接到 doc，过滤语言
        q = (
            s.query(DocChunkORM.id, DocChunkORM.text)
            .join(DocORM, DocChunkORM.doc_id == DocORM.id)
            .filter(DocORM.lang == lang)
            .filter(DocChunkORM.tier.in_(tiers_tuple))
        )
        # 排序：最近写入的在前（可按需要改）
        q = q.order_by(DocChunkORM.id.desc())
        rows = q.limit(limit*2).all()  # 预抓宽一点，后面再减去已存在的
        cand = [(int(cid), txt or "") for (cid, txt) in rows]

    if rebuild:
        return cand[:limit]

    # 过滤掉已嵌入（vec_chunk 已有该 model 的）
    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()
    exists = set()
    for (cid,) in cur.execute("SELECT chunk_id FROM vec_chunk WHERE model=?", (model_name,)):
        exists.add(int(cid))
    con.close()
    out = [(cid, txt) for (cid, txt) in cand if cid not in exists]
    return out[:limit]

def upsert_vec_chunk(chunk_ids:List[int], model_name:str, dim:int, norms:np.ndarray):
    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()
    cur.execute("BEGIN")
    for cid, nrm in zip(chunk_ids, norms):
        # INSERT OR REPLACE：保证幂等
        cur.execute(
            "INSERT OR REPLACE INTO vec_chunk(chunk_id, model, dim, norm) VALUES(?,?,?,?)",
            (int(cid), model_name, int(dim), float(nrm))
        )
    con.commit(); con.close()

def sync_fts_chunk(chunk_rows: List[Tuple[int,str]]):
    # Be resilient across contentless/contentful FTS5 tables:
    # 1) Try special delete-token (contentless/external)
    # 2) If it fails, fall back to regular DELETE (contentful)
    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()
    cur.execute("BEGIN")
    try:
        for cid, txt in chunk_rows:
            ok = False
            try:
                cur.execute(
                    "INSERT INTO fts_chunk(fts_chunk, rowid, text) VALUES('delete', ?, NULL)",
                    (int(cid),),
                )
                ok = True
            except sqlite3.OperationalError:
                # fall back for contentful FTS5
                cur.execute("DELETE FROM fts_chunk WHERE rowid=?", (int(cid),))
            # insert latest text
            cur.execute("INSERT INTO fts_chunk(rowid, text) VALUES(?, ?)", (int(cid), txt))
        con.commit()
    finally:
        con.close()

# ------------------ FAISS helpers ------------------
def build_or_append_faiss(embedder:Embedder, ids:np.ndarray, vecs:np.ndarray, args):
    if not FAISS_OK:
        return

    ids = ids.astype("int64")
    dim = int(embedder.dim())

    if args.rebuild or (not INDEX_PATH.exists()):
        logger.info("Building new FAISS HNSW index ...")
        base = faiss.IndexHNSWFlat(dim, args.hnsw_M)
        base.hnsw.efConstruction = args.hnsw_efC
        base.hnsw.efSearch = args.hnsw_efS
        index = faiss.IndexIDMap2(base)
        index.add_with_ids(vecs, ids)
    else:
        logger.info("Appending to existing FAISS index ...")
        index = faiss.read_index(str(INDEX_PATH))
        # 确保是带 ID 的；若不是，则包一层（理论上我们一直用 IDMap2）
        if not isinstance(index, faiss.IndexIDMap2):
            index = faiss.IndexIDMap2(index)
        index.add_with_ids(vecs, ids)

    # 写盘
    faiss.write_index(index, str(INDEX_PATH))
    meta = {
        "model": embedder.name(),
        "dim": dim,
        "ntotal": int(index.ntotal),
        "index_path": str(INDEX_PATH),
        "type": "HNSWFlat+IDMap2",
        "hnsw": {"M": args.hnsw_M, "efConstruction": args.hnsw_efC, "efSearch": args.hnsw_efS},
        "updated_at": int(time.time())
    }
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"FAISS saved: ntotal={meta['ntotal']}")

# ------------------ main ------------------
def main():
    args = parse_args()
    # Ensure FTS table exists; when --rebuild, recreate it cleanly
    ensure_fts_chunk(args.rebuild)
    embedder = make_embedder(args)
    model_tag = embedder.name()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    cand = select_candidate_chunks(args.limit, args.lang, tiers, model_tag, args.rebuild)
    logger.info(f"candidates={len(cand)} (lang={args.lang}, tiers={tiers}, model={model_tag})")
    if not cand:
        logger.info("Nothing to embed.")
        return

    # 批量编码
    chunk_ids, texts = zip(*cand)
    vecs = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i+args.batch_size]
        v = embedder.encode(list(batch))
        vecs.append(v)
    vecs = np.vstack(vecs).astype("float32")
    # 单位向量，norm 近似 1；保存 norm 以便监控
    norms = np.linalg.norm(vecs, axis=1)

    # 写 vec_chunk + fts_chunk
    upsert_vec_chunk(list(chunk_ids), model_tag, embedder.dim(), norms)
    sync_fts_chunk(list(zip(chunk_ids, texts)))

    # 构建/追加 FAISS
    build_or_append_faiss(embedder, np.array(chunk_ids), vecs, args)

    logger.info("done.")

if __name__ == "__main__":
    main()