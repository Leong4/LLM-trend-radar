# data_ingestion_skeleton.py
"""
Async ingestion core for LLM Trend Radar (POC)
------------------------------------------------
Contains:
    • async_arxiv.py  – pull papers & PDFs via arXiv API
    • ingest_runner.py – orchestration entry‑point with Rich progress
You will grow this into a package, but start with a single file for rapid iteration.
"""

from __future__ import annotations
import asyncio
import datetime as dt
import hashlib
import os
import pathlib
from typing import List

SEM_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_KEY")  # set in .env if you have one

# ---------- Path configuration (robust for external drives) ----------
ROOT_DIR = pathlib.Path(__file__).resolve().parent          # /Volumes/…/project1
DATA_DIR = ROOT_DIR / "data"
PDF_DIR  = DATA_DIR / "papers_pdf"
LOG_DIR  = ROOT_DIR / "logs"

# Create directories if they don't already exist
for _dir in (DATA_DIR, PDF_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

DB_FILE = DATA_DIR / "trend_radar.db"
DB_PATH = f"sqlite:///{DB_FILE}"
# ---------------------------------------------------------------------

import httpx
from loguru import logger
from pydantic import BaseModel, Field
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sqlalchemy import Column, DateTime, Integer, String, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

# ---------------------------
# CONFIG – will later move to TOML
# ---------------------------
STATS_FILE = DATA_DIR / "stats.json"

ARXIV_CATEGORY = "cs.CL"
ARXIV_MAX_RESULTS = 100  # daily fetch upper‑bound
ARXIV_QUERY_WINDOW_DAYS = 7
CITATION_THRESHOLD = 0  # Semantic Scholar minimal citations
# If no Semantic Scholar key is provided, citation look‑ups may fail/return None; set threshold to 0 or use --no-citation-filter flag in that case.
CONCURRENT_REQUESTS = 10  # async semaphore

# GitHub / HuggingFace / npm token placeholders (set via env)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# ---------------------------
# DB setup
# ---------------------------
Base = declarative_base()


class PaperORM(Base):
    __tablename__ = "paper"
    id = Column(Integer, primary_key=True)
    arxiv_id = Column(String, unique=True)
    title = Column(String)
    authors = Column(String)
    published = Column(DateTime)
    citations = Column(Integer)
    pdf_path = Column(String)
    status = Column(String, default="gray")  # gray | full
    last_checked = Column(DateTime, default=dt.datetime.utcnow)
    gray_reason  = Column(String, nullable=True)
    source_type = Column(String, default="arxiv")
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    lang = Column(String, nullable=True)
    fingerprint = Column(String, unique=True)


engine = create_engine(DB_PATH, echo=False, future=True)
Base.metadata.create_all(engine)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# ---------------------------
# UTILS
# ---------------------------

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def iso_date_minus(days: int) -> str:
    return (dt.datetime.utcnow() - dt.timedelta(days=days)).strftime("%Y%m%d")


# ---------------------------
# Pydantic schema (interface contract v0)
# ---------------------------
class PaperSchema(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str]
    published: dt.datetime
    citations: int | None = None
    pdf_url: str | None = None
    pdf_path: str | None = None
    status: str = "gray"  # gray/full
    last_checked: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    gray_reason: str | None = None
    source_type: str = "arxiv"
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    lang: str | None = None

    class Config:
        arbitrary_types_allowed = True

# alias for backward compatibility with existing code
Paper = PaperSchema


# ---------------------------
# Async fetch functions
# ---------------------------
ARXIV_API = "https://export.arxiv.org/api/query"
SEM_SCHOLAR_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}?fields=citationCount"


async def fetch_url(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url, timeout=30)
    r.raise_for_status()
    return r.text


async def fetch_citations(client: httpx.AsyncClient, arxiv_id: str) -> int | None:
    url = SEM_SCHOLAR_ENDPOINT.format(arxiv_id=arxiv_id)
    try:
        headers = {"x-api-key": SEM_SCHOLAR_API_KEY} if SEM_SCHOLAR_API_KEY else None
        data_resp = await client.get(url, headers=headers, timeout=20)
        data_resp.raise_for_status()
        data = data_resp.json()
        return int(data.get("citationCount", 0))
    except Exception as e:
        logger.warning(f"Semantic Scholar lookup failed for {arxiv_id}: {e}")
        return None
        pass


async def download_pdf(client: httpx.AsyncClient, paper: Paper) -> str | None:
    if not paper.pdf_url:
        return None
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    fname = PDF_DIR / f"{paper.arxiv_id}.pdf"
    try:
        resp = await client.get(paper.pdf_url, timeout=60)
        resp.raise_for_status()
        fname.write_bytes(resp.content)
        return str(fname)
    except Exception as e:
        logger.error(f"PDF download failed for {paper.arxiv_id}: {e}")
        return None

async def fetch_github_batch() -> List[Paper]:
    """Fetch trending GitHub repos (topic: rag) via REST API search."""
    headers = {
        "Accept": "application/vnd.github+json",
        **({"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}),
    }
    params = {
        "q": "topic:rag stars:>0",
        "sort": "stars",
        "order": "desc",
        "per_page": 30,
    }
    url = "https://api.github.com/search/repositories"
    async with httpx.AsyncClient(headers=headers, timeout=30) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"GitHub fetch failed: {e}")
            return []
    papers: List[Paper] = []
    for item in data.get("items", []):
        papers.append(
            Paper(
                arxiv_id=f"github:{item['id']}",          # repo id as surrogate key
                title=item["full_name"],
                authors=[item["owner"]["login"]],
                published=dt.datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                citations=item["stargazers_count"],
                pdf_url=None,
                status="gray",
                source_type="github",
            )
        )
    logger.info(f"GitHub items pulled: {len(papers)}")
    return papers

async def fetch_arxiv_batch() -> List[Paper]:
    # form search query
    query = f"cat:{ARXIV_CATEGORY}"  # pull latest first; we'll filter by date in code
    params = {
        "search_query": query,
        "start": 0,
        "max_results": ARXIV_MAX_RESULTS,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    async with httpx.AsyncClient() as client:
        logger.info("Querying arXiv API…")
        query_str = str(httpx.QueryParams(params))  # httpx.QueryParams → URL‑encoded str
        feed_xml = await fetch_url(client, f"{ARXIV_API}?{query_str}")
        import feedparser  # local import to keep global deps minimal

        feed = feedparser.parse(feed_xml)
        tasks = []
        papers: List[Paper] = []
        sem = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def handle_entry(entry):
            async with sem:
                arxiv_id = entry.id.split("/abs/")[1]
                citations = await fetch_citations(client, arxiv_id)
                paper = Paper(
                    arxiv_id=arxiv_id,
                    title=entry.title.strip().replace("\n", " "),
                    authors=[a.name for a in entry.authors],
                    published=dt.datetime(*entry.published_parsed[:6]),
                    citations=citations,
                    pdf_url=next((l.href for l in entry.links if l.type == "application/pdf"), None),
                )
                papers.append(paper)

        for entry in feed.entries:
            tasks.append(asyncio.create_task(handle_entry(entry)))

        await asyncio.gather(*tasks)

        # --- time window filter --------------------------------------
        cutoff_dt = dt.datetime.utcnow() - dt.timedelta(days=ARXIV_QUERY_WINDOW_DAYS)
        papers = [p for p in papers if p.published >= cutoff_dt]
        logger.info(f"Retrieved {len(papers)} papers within {ARXIV_QUERY_WINDOW_DAYS}‑day window")

        # filter by citation threshold
        qualified = [p for p in papers if (p.citations or 0) >= CITATION_THRESHOLD]
        # mark status gray
        logger.info(f"Qualified after citation filter: {len(qualified)}")
        return qualified


async def enrich_and_save(papers: List[Paper]):
    async with httpx.AsyncClient() as client:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            transient=True,
        )
        progress.start()
        task = progress.add_task("Downloading PDFs", total=len(papers))
        sem = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def process(p: Paper):
            async with sem:
                p.pdf_path = await download_pdf(client, p)
                p.status = "full" if p.pdf_path else "gray"
                fp = sha1(p.title)
                orm = PaperORM(
                    arxiv_id=p.arxiv_id,
                    title=p.title,
                    authors=", ".join(p.authors),
                    published=p.published,
                    citations=p.citations or 0,
                    pdf_path=p.pdf_path,
                    status=p.status,
                    source_type=p.source_type,
                    created_at=dt.datetime.utcnow(),
                    lang=p.lang,
                    fingerprint=fp,
                )
                with SessionLocal() as session:
                    try:
                        session.add(orm)
                        session.commit()
                    except IntegrityError:
                        session.rollback()
                progress.advance(task)

        await asyncio.gather(*(process(p) for p in papers))
        progress.stop()


# ---------------------------
# Gray‑list promotion helper
# ---------------------------
async def promote_gray_items(days: int = 14):
    """
    Re‑check gray items older than *days*; promote to 'full_candidate' or mark stale.
    Promotion rule:
      * github  : stars ≥ 50
      * arxiv   : citations ≥ 5
    """
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=days)
    with SessionLocal() as s:
        gray_items = s.scalars(
            select(PaperORM).where(
                PaperORM.status == "gray",
                PaperORM.last_checked < cutoff,
            )
        ).all()

    if not gray_items:
        logger.info("Gray promoter: no items to re‑check")
        return

    async with httpx.AsyncClient(timeout=20) as cli:
        for it in gray_items:
            try:
                if it.source_type == "github":
                    repo_id = it.arxiv_id.split(":")[1]
                    gh_url = f"https://api.github.com/repositories/{repo_id}"
                    headers = {
                        "Accept": "application/vnd.github+json",
                        **({"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}),
                    }
                    stars = (await cli.get(gh_url, headers=headers)).json().get("stargazers_count", 0)
                    if stars >= 50:
                        it.status = "full_candidate"
                        it.citations = stars
                    else:
                        it.gray_reason = f"star={stars}"
                elif it.source_type == "arxiv":
                    cit = await fetch_citations(cli, it.arxiv_id)
                    if cit and cit >= 5:
                        it.status = "full_candidate"
                        it.citations = cit
                    else:
                        it.gray_reason = f"cit={cit}"
                it.last_checked = dt.datetime.utcnow()
            except Exception as e:
                logger.warning(f"Gray promoter error {it.arxiv_id}: {e}")

        with SessionLocal() as s:
            s.commit()
    logger.info(f"Gray promoter processed {len(gray_items)} records")


def adjust_arxiv_window(new_count: int) -> int:
    """Adjust ARXIV_QUERY_WINDOW_DAYS based on last fetch volume."""
    history = []
    if STATS_FILE.exists():
        import json
        history = json.loads(STATS_FILE.read_text())
    history.append({"ts": dt.datetime.utcnow().isoformat(), "new": new_count})
    history = history[-7:]  # keep last 7 records
    import json, statistics
    STATS_FILE.write_text(json.dumps(history, indent=2))

    avg = statistics.mean(h["new"] for h in history)
    wnd = ARXIV_QUERY_WINDOW_DAYS
    if avg < 20:
        wnd = min(wnd + 2, 14)
    elif avg > 100:
        wnd = max(wnd - 1, 3)
    return wnd


# ---------------------------
# Runner CLI
# ---------------------------

def main():
    logger.add(str(LOG_DIR / "ingest_{time}.log"), rotation="10 MB")
    logger.info("=== LLM Trend Radar ingest run ===")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(promote_gray_items())
    arxiv, gh = loop.run_until_complete(
        asyncio.gather(
            fetch_arxiv_batch(),
            fetch_github_batch(),
        )
    )
    papers = arxiv + gh
    logger.info(f"Debug counts → arxiv:{len(arxiv)} gh:{len(gh)}")
    # dynamic window adjust
    global ARXIV_QUERY_WINDOW_DAYS
    ARXIV_QUERY_WINDOW_DAYS = adjust_arxiv_window(len(arxiv))
    logger.info(f"Next arXiv window will be {ARXIV_QUERY_WINDOW_DAYS} days")
    loop.run_until_complete(enrich_and_save(papers))
    logger.info("Run complete")


if __name__ == "__main__":
    main()
