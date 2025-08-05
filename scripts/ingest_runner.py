#!/usr/bin/env python3
import argparse, asyncio
from loguru import logger
from llm_radar import config
from llm_radar.fetch import arxiv, github
from llm_radar.pipeline import ingest, promoter, window

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=config.ARXIV_QUERY_WINDOW_DAYS)
    ap.add_argument("--download-pdf", action="store_true")
    return ap.parse_args()

def main():
    args = cli()
    config.ARXIV_QUERY_WINDOW_DAYS = args.days

    logger.add(config.LOG_DIR / "runner_{time}.log", rotation="10 MB")
    loop = asyncio.get_event_loop()

    loop.run_until_complete(promoter.promote_gray_items())

    arx, gh = loop.run_until_complete(asyncio.gather(
        arxiv.fetch_arxiv_batch(),
        github.fetch_github_batch(),
    ))
    papers = arx + gh
    logger.info(f"Debug → arxiv:{len(arx)} gh:{len(gh)}")

    config.ARXIV_QUERY_WINDOW_DAYS = window.adjust_arxiv_window(len(arx))

    if args.download_pdf:
        loop.run_until_complete(ingest.enrich_and_save(papers))
    else:
        loop.run_until_complete(ingest.enrich_and_save([p for p in papers if p.source_type=="github"]))

    logger.info("Run complete ✅")

if __name__ == "__main__":
    main()