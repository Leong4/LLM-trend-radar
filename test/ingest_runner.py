"""ingest_runner.py
===============
统一调度脚本：调用 data_ingestion_skeleton 中的 fetch & save 方法，
可通过 CLI 参数控制时间窗口 / 是否下载 PDF，方便后续 cron 或 GitHub-Actions 调用。
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
from pathlib import Path
from loguru import logger

# 动态导入 skeleton，避免循环引用
skeleton = importlib.import_module("data_ingestion_skeleton")


def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM Trend Radar – Ingestion Runner")
    ap.add_argument("--days", type=int, default=skeleton.ARXIV_QUERY_WINDOW_DAYS,
                    help="Look-back window for arXiv query (default same as skeleton)")
    ap.add_argument("--download-pdf", action="store_true",
                    help="Download PDFs for qualified arXiv papers")
    return ap.parse_args()


def main():
    args = parse_cli()

    # 动态覆盖 skeleton 的查询窗口
    skeleton.ARXIV_QUERY_WINDOW_DAYS = args.days

    logger.info("=== Ingest Runner start ===")
    # Step 0: promote gray items before new fetch
    loop = asyncio.get_event_loop()
    loop.run_until_complete(skeleton.promote_gray_items())

    arxiv, github = loop.run_until_complete(
        asyncio.gather(
            skeleton.fetch_arxiv_batch(),
            skeleton.fetch_github_batch(),
        )
    )
    papers = arxiv + github
    logger.info(f"Debug counts → arxiv:{len(arxiv)} github:{len(github)}")
    # adjust arXiv query window for next run based on today’s new count
    skeleton.ARXIV_QUERY_WINDOW_DAYS = skeleton.adjust_arxiv_window(len(arxiv))
    logger.info(f"Next arXiv window will be {skeleton.ARXIV_QUERY_WINDOW_DAYS} days")

    if args.download_pdf:
        loop.run_until_complete(skeleton.enrich_and_save(papers))
    else:
        # Save all metadata but skip heavy PDF downloads
        loop.run_until_complete(
            skeleton.enrich_and_save([p for p in papers if p.source_type == "github"])
        )

    logger.info("Run complete")


if __name__ == "__main__":
    main()
