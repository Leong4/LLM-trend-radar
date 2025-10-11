#!/usr/bin/env python3
import argparse
import asyncio
import sqlite3
import json
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

from llm_radar.db.models import SessionLocal, PaperORM, DocORM, DocChunkORM
from llm_radar.preprocess.lite import preprocess_arxiv_lite, preprocess_github_lite
from llm_radar.config import DATA_DIR

from llm_radar.utils.conf import load_config

import re
import tempfile
import httpx

try:
    from pdfminer.high_level import extract_text as _pdf_extract_text  # type: ignore
except Exception:  # pdfminer not installed or runtime issue
    _pdf_extract_text = None

DB_PATH = Path(DATA_DIR) / "trend_radar.db"

# ---- config-overridable defaults ----
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
STRIP_REFS = True
PDF_BACKFILL = False
DEFAULT_UCB_THRESH = 0.70

# ----------------------- Lang detection (graceful) -----------------------
try:
    from langdetect import detect as _ld_detect  # type: ignore
    def detect_lang(text: str) -> str:
        try:
            return _ld_detect(text or "")
        except Exception:
            return heuristic_lang(text)
except Exception:
    def detect_lang(text: str) -> str:
        return heuristic_lang(text)

def heuristic_lang(text: str) -> str:
    if not text:
        return "und"
    # very rough: Chinese vs English vs other
    zh = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    ascii_letters = sum(1 for ch in text if ('a' <= ch <= 'z') or ('A' <= ch <= 'Z'))
    if zh > 10 and zh > ascii_letters:
        return "zh"
    if ascii_letters > 10 and ascii_letters >= zh:
        return "en"
    return "und"

def apply_config(cfg: dict | None):
    """Apply YAML config for preprocess runner."""
    if not cfg:
        return
    try:
        PP = (cfg or {}).get("preprocess", {})
        pdf = PP.get("pdf", {})
        global CHUNK_SIZE, CHUNK_OVERLAP, STRIP_REFS, DEFAULT_UCB_THRESH
        CHUNK_SIZE = int(PP.get("chunk_size", CHUNK_SIZE))
        CHUNK_OVERLAP = int(PP.get("overlap", CHUNK_OVERLAP))
        STRIP_REFS = bool(pdf.get("strip_references", STRIP_REFS))
        DEFAULT_UCB_THRESH = float(PP.get("ucb_threshold", DEFAULT_UCB_THRESH))
    except Exception as e:
        logger.warning(f"apply_config failed: {e}")

# ----------------------- PDF parse helpers -----------------------
def strip_references(text: str) -> str:
    """Remove References section heuristically: cut from a line that equals 'References' (case-insensitive)
    or 'Bibliography'/'Acknowledgements'."""
    if not text:
        return text
    # Normalize line endings
    t = text.replace('\r\n', '\n').replace('\r', '\n')
    # Find the first heading-like occurrence
    pat = re.compile(r'\n\s*(references|bibliography|acknowledgements?)\s*\n', re.IGNORECASE)
    m = pat.search(t)
    return t[:m.start()] if m else t

def clean_text(t: str) -> str:
    if not t:
        return ''
    t = re.sub(r"[\t\x0b\x0c]+", " ", t)
    t = re.sub(r"\u00a0", " ", t)  # nbsp
    # collapse long spaces
    t = re.sub(r"\s{2,}", " ", t)
    # restore newlines for paragraphs
    t = re.sub(r"\s*\n\s*", "\n", t)
    return t.strip()

def chunk_text(t: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    t = t.strip()
    if not t:
        return []
    chunks = []
    i = 0
    n = len(t)
    step = max(1, chunk_size - overlap)
    while i < n:
        chunks.append(t[i:i+chunk_size])
        i += step
    return chunks

async def preprocess_arxiv_pdf_first(p: PaperORM):
    """Try PDF-first parse for arXiv; fallback to lite (abstract) if PDF unavailable.
    Returns (doc_like, chunk_likes).
    """
    ax_id = getattr(p, 'arxiv_id', None)
    if not ax_id:
        raise RuntimeError('arXiv id missing on PaperORM')
    # If pdfminer not available, bail to lite by raising
    if _pdf_extract_text is None:
        raise RuntimeError('pdfminer not available')
    url = f"https://arxiv.org/pdf/{ax_id}.pdf"
    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as cli:
        r = await cli.get(url)
        if r.status_code != 200 or not r.content:
            raise RuntimeError(f'pdf download failed: {r.status_code}')
        # write to temp file for pdfminer
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True) as fp:
            fp.write(r.content)
            fp.flush()
            try:
                raw_text = _pdf_extract_text(fp.name) or ''
            except Exception as e:
                raise RuntimeError(f'pdf parse failed: {e}')
    text = clean_text(strip_references(raw_text) if STRIP_REFS else raw_text)
    if len(text) < 400:  # too short, likely bad parse
        raise RuntimeError('pdf text too short')
    # Build pseudo doc & chunks
    class _Doc: pass
    class _Chunk: pass
    doc = _Doc()
    doc.lang = None
    doc.n_pages = None
    doc.parse_status = 'pdf_ok'
    doc.parse_error = None
    doc.process_stage = 'pdf'
    chunks_out = []
    for s in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
        ch = _Chunk()
        ch.text = s
        ch.section_title = None
        ch.n_chars = len(s)
        ch.n_tokens = None
        ch.lang = None
        ch.page_start = None
        ch.page_end = None
        ch.tier = None
        ch.hash = None
        chunks_out.append(ch)
    return doc, chunks_out

# ----------------------- Candidate selection -----------------------

def select_candidates(limit: int, ucb_threshold: float) -> list[Tuple[str, str, str]]:
    """
    从 items 表选择要预处理的对象：
      - FULL 全量；
      - PROBATION 且 last_ucb >= 阈值（高探索价值）。
    返回 [(item_id, source, state), ...]，按 FULL 优先、last_ucb 降序。
    若 items 表不存在，则回退到旧逻辑（paper.status = 'gray'）。
    """
    if not DB_PATH.exists():
        logger.warning(f"DB not found at {DB_PATH}; fallback to legacy PaperORM selection")
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='items'")
        if not cur.fetchone():
            logger.warning("items table not found; fallback to legacy PaperORM selection")
            return []
        rows = cur.execute(
            """
            SELECT id, source, state
            FROM items
            WHERE state='FULL' OR (state='PROBATION' AND last_ucb >= ?)
            ORDER BY (state='FULL') DESC, last_ucb DESC
            LIMIT ?
            """,
            (ucb_threshold, limit)
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]
    finally:
        try:
            conn.close()
        except Exception:
            pass


def map_item_to_paper(session: SessionLocal, item_id: str, source: str) -> Optional[PaperORM]:
    """将 items.id 映射回 PaperORM 记录。
       - arXiv: items.id = 'ax:<arxiv_id>' → PaperORM.arxiv_id
       - GitHub: items.id = 'gh:owner/repo' → 尝试按 title 包含 owner/repo 匹配
    """
    try:
        if source == "arxiv" and item_id.startswith("ax:"):
            ax_id = item_id.split(":", 1)[1]
            return session.query(PaperORM).filter(
                PaperORM.source_type == "arxiv",
                PaperORM.arxiv_id == ax_id
            ).first()
        if source == "github" and item_id.startswith("gh:"):
            owner_repo = item_id.split(":", 1)[1]
            # 优先完整匹配，其次仅 repo 名匹配
            q = session.query(PaperORM).filter(PaperORM.source_type == "github")
            row = q.filter(PaperORM.title.ilike(f"%{owner_repo}%")).first()
            if row:
                return row
            repo_only = owner_repo.split("/")[-1]
            return q.filter(PaperORM.title.ilike(f"%{repo_only}%")).first()
    except Exception as e:
        logger.warning(f"map_item_to_paper failed for {item_id} ({source}): {e}")
    return None

# ----------------------- CLI & Runner -----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Preprocess runner (Lite/Full)")
    ap.add_argument("--mode", choices=["lite", "full"], default="lite")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--github-token", type=str, default=None)
    ap.add_argument("--ucb-threshold", type=float, default=None, help="PROBATION 的 UCB 选择阈值 (CLI 覆盖 YAML)")
    ap.add_argument("--pdf-backfill", action="store_true", help="将 arXiv 的 lite 文档回填为 PDF 解析并覆盖 chunks")
    ap.add_argument("--config", type=str, default=None, help="path to config/promoter.yaml")
    return ap.parse_args()


async def run_lite(limit: int, gh_token: str | None, ucb_threshold: float, pdf_backfill: bool):
    # 1) 按新状态机选择候选（优先 FULL，再选高 UCB 的 PROBATION）
    candidates = select_candidates(limit, ucb_threshold)

    # 若 items 不可用，则回退到旧逻辑（paper.status = 'gray'，仅用于保底）
    legacy_fallback = False
    if not candidates:
        legacy_fallback = True
        with SessionLocal() as s:
            existing = {pid for (pid,) in s.query(DocORM.paper_id).all()}
            papers = s.query(PaperORM).filter(PaperORM.status == "gray").all()
            candidates_papers = [p for p in papers if p.id not in existing][:limit]

    total, ok, fail = 0, 0, 0

    # 2) 逐个处理
    if legacy_fallback:
        for p in candidates_papers:
            total += 1
            tier = "probation-lite"
            await _process_one(p, tier=tier, gh_token=gh_token)
            ok += 1  # _process_one 内部会记录异常统计
    else:
        with SessionLocal() as s:
            for item_id, source, state in candidates:
                total += 1
                p = map_item_to_paper(s, item_id, source)
                if not p:
                    logger.warning(f"skip: cannot map item {item_id} -> PaperORM")
                    continue
                # 处理已存在 doc：若开启 --pdf-backfill 且为 arXiv 的 lite，则尝试覆盖为 PDF
                drow_exist = s.query(DocORM).filter(DocORM.paper_id == p.id).first()
                if drow_exist:
                    if pdf_backfill and p.source_type == "arxiv" and (drow_exist.process_stage or "lite") == "lite":
                        try:
                            doc_new, chunks = await preprocess_arxiv_pdf_first(p)
                        except Exception as e:
                            logger.info(f"pdf-backfill failed for paper_id={p.id}: {e}")
                            continue
                        # 语言补全
                        if not getattr(doc_new, "lang", None):
                            sample = "\n".join((ch.text or "")[:800] for ch in (chunks[:3] if chunks else []))
                            doc_new.lang = detect_lang(sample)
                        # 覆盖 doc 字段
                        drow_exist.lang = doc_new.lang
                        drow_exist.parse_status = getattr(doc_new, "parse_status", "pdf_ok")
                        drow_exist.parse_error = None
                        drow_exist.process_stage = getattr(doc_new, "process_stage", "pdf")
                        # 删除旧 chunks，再写入新 chunks
                        s.query(DocChunkORM).filter(DocChunkORM.doc_id == drow_exist.id).delete()
                        for i, ch in enumerate(chunks):
                            s.add(DocChunkORM(
                                doc_id=drow_exist.id,
                                section_title=getattr(ch, "section_title", None),
                                chunk_index=i,
                                text=ch.text,
                                n_chars=getattr(ch, "n_chars", len(ch.text or "")),
                                n_tokens=getattr(ch, "n_tokens", None),
                                lang=getattr(ch, "lang", None),
                                page_start=getattr(ch, "page_start", None),
                                page_end=getattr(ch, "page_end", None),
                                tier="full" if state == "FULL" else "probation-lite",
                                hash=getattr(ch, "hash", None),
                            ))
                        s.commit()
                        logger.info(f"pdf-backfill success for paper_id={p.id}, doc_id={drow_exist.id}")
                        ok += 1
                        continue
                    else:
                        logger.info(f"skip existing doc for paper_id={p.id}")
                        continue
                tier = "full" if state == "FULL" else "probation-lite"
                try:
                    await _process_one(p, tier=tier, gh_token=gh_token)
                    ok += 1
                except Exception:
                    fail += 1

    logger.info(f"Lite done: total={total} ok={ok} fail={fail}")
    # 3) 简单导出统计
    try:
        from datetime import date
        exp = Path("exports"); exp.mkdir(exist_ok=True)
        out = exp / "preprocess_stats.csv"
        hdr = not out.exists()
        with out.open("a", encoding="utf-8") as f:
            if hdr:
                f.write("date,mode,total,ok,fail\n")
            f.write(f"{date.today().isoformat()},lite,{total},{ok},{fail}\n")
    except Exception:
        pass


async def _process_one(p: PaperORM, tier: str, gh_token: Optional[str]):
    try:
        # 运行轻预处理（抓文本 + 清洗 + 切块）
        if p.source_type == "arxiv":
            try:
                doc, chunks = await preprocess_arxiv_pdf_first(p)  # PDF 优先，自动跳过 References
            except Exception as _e_pdf:
                logger.info(f"PDF-first failed, fallback to abstract: {_e_pdf}")
                doc, chunks = await preprocess_arxiv_lite(p)
        elif p.source_type == "github":
            doc, chunks = await preprocess_github_lite(p, token=gh_token)
        else:
            logger.info(f"skip unsupported source: {p.source_type}")
            return

        # 语言补全（若上游未提供）
        if not getattr(doc, "lang", None):
            sample = "\n".join((ch.text or "")[:800] for ch in (chunks[:3] if chunks else []))
            doc.lang = detect_lang(sample)
        for ch in chunks:
            if not getattr(ch, "lang", None):
                ch.lang = detect_lang(ch.text or "")
            # 以状态机分层决定 tier（若上游未给）
            if not getattr(ch, "tier", None):
                ch.tier = tier

        # 写入数据库
        with SessionLocal() as s:
            drow = DocORM(
                paper_id=p.id,
                source_type=p.source_type,
                title=p.title,
                authors=p.authors,
                published=p.published,
                lang=doc.lang,
                n_pages=getattr(doc, "n_pages", None),
                parse_status=getattr(doc, "parse_status", None),
                parse_error=getattr(doc, "parse_error", None),
                process_stage=getattr(doc, "process_stage", "lite"),
            )
            s.add(drow); s.commit(); s.refresh(drow)

            for i, ch in enumerate(chunks):
                s.add(DocChunkORM(
                    doc_id=drow.id,
                    section_title=getattr(ch, "section_title", None),
                    chunk_index=i,
                    text=ch.text,
                    n_chars=getattr(ch, "n_chars", len(ch.text or "")),
                    n_tokens=getattr(ch, "n_tokens", None),
                    lang=getattr(ch, "lang", None),
                    page_start=getattr(ch, "page_start", None),
                    page_end=getattr(ch, "page_end", None),
                    tier=getattr(ch, "tier", tier),
                    hash=getattr(ch, "hash", None),
                ))
            s.commit()
    except Exception as e:
        logger.exception(f"Preprocess failed for {p.source_type}:{getattr(p,'arxiv_id',None) or p.title} — {e}")
        raise


async def main():
    args = parse_args()
    # load & apply YAML config (safe fallback to defaults)
    try:
        cfg = load_config(args.config)
    except Exception:
        cfg = None
    apply_config(cfg)

    # resolve UCB threshold: CLI overrides YAML; YAML overrides hard default
    ucb_thr = args.ucb_threshold if args.ucb_threshold is not None else DEFAULT_UCB_THRESH

    if args.mode == "lite":
        await run_lite(args.limit, args.github_token, ucb_thr, args.pdf_backfill)
    else:
        logger.warning("Full mode not implemented yet; run with --mode lite")

if __name__ == "__main__":
    asyncio.run(main())