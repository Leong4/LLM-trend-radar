#!/usr/bin/env python3
import argparse
import asyncio
from loguru import logger

from llm_radar.db.models import SessionLocal, PaperORM, DocORM, DocChunkORM
from llm_radar.preprocess.lite import preprocess_arxiv_lite, preprocess_github_lite

def parse_args():
    ap = argparse.ArgumentParser(description="Preprocess runner (Lite/Full)")
    ap.add_argument("--mode", choices=["lite", "full"], default="lite")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--github-token", type=str, default=None)
    return ap.parse_args()

async def run_lite(limit: int, gh_token: str | None):
    with SessionLocal() as s:
        existing = {pid for (pid,) in s.query(DocORM.paper_id).all()}
        papers = s.query(PaperORM).filter(PaperORM.status == "gray").all()
        candidates = [p for p in papers if p.id not in existing][:limit]

    total, ok, fail = 0, 0, 0
    for p in candidates:
        total += 1
        try:
            if p.source_type == "arxiv":
                doc, chunks = await preprocess_arxiv_lite(p)
            elif p.source_type == "github":
                doc, chunks = await preprocess_github_lite(p, token=gh_token)
            else:
                continue

            with SessionLocal() as s:
                drow = DocORM(
                    paper_id=p.id,
                    source_type=p.source_type,
                    title=p.title,
                    authors=p.authors,
                    published=p.published,
                    lang=doc.lang,
                    n_pages=doc.n_pages,
                    parse_status=doc.parse_status,
                    parse_error=doc.parse_error,
                    process_stage=doc.process_stage,
                )
                s.add(drow); s.commit(); s.refresh(drow)
                for i, ch in enumerate(chunks):
                    s.add(DocChunkORM(
                        doc_id=drow.id,
                        section_title=ch.section_title,
                        chunk_index=i,
                        text=ch.text,
                        n_chars=ch.n_chars,
                        n_tokens=ch.n_tokens,
                        lang=ch.lang,
                        page_start=ch.page_start,
                        page_end=ch.page_end,
                        tier=ch.tier,
                        hash=ch.hash,
                    ))
                s.commit()
            ok += 1
        except Exception as e:
            logger.exception(f"Preprocess failed for {p.source_type}:{p.arxiv_id} — {e}")
            fail += 1
    logger.info(f"Lite done: total={total} ok={ok} fail={fail}")

async def main():
    args = parse_args()
    if args.mode == "lite":
        await run_lite(args.limit, args.github_token)
    else:
        logger.warning("Full mode not implemented yet; run with --mode lite")

if __name__ == "__main__":
    asyncio.run(main())