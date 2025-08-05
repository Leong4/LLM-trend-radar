import asyncio, datetime as dt, httpx, feedparser
from loguru import logger

# Constants specific to arXiv fetching
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_MAX_RESULTS = 100  # maximum records to pull per request

from ..config import (
    SEM_SCHOLAR_API_KEY,
    ARXIV_CATEGORY,
    ARXIV_QUERY_WINDOW_DAYS,
    CITATION_THRESHOLD,
    CONCURRENT_REQUESTS,
)
from ..db.schema import PaperSchema as Paper

SEM_SCH_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}?fields=citationCount"

async def fetch_citations(client: httpx.AsyncClient, arxiv_id: str) -> int | None:
    url = SEM_SCH_ENDPOINT.format(arxiv_id=arxiv_id)
    headers = {"x-api-key": SEM_SCHOLAR_API_KEY} if SEM_SCHOLAR_API_KEY else None
    try:
        resp = await client.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        return int(resp.json().get("citationCount", 0))
    except Exception:
        return None

async def fetch_arxiv_batch() -> list[Paper]:
    query = f"cat:{ARXIV_CATEGORY}"
    params = {"search_query": query, "start": 0, "max_results": ARXIV_MAX_RESULTS,
              "sortBy": "submittedDate", "sortOrder": "descending"}
    async with httpx.AsyncClient() as client:
        feed_xml = (await client.get(ARXIV_API, params=params)).text
    feed = feedparser.parse(feed_xml)

    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
    async def handle(entry):
        async with sem:
            arxiv_id = entry.id.split("/abs/")[1]
            cit = await fetch_citations(client, arxiv_id)
            return Paper(
                arxiv_id=arxiv_id,
                title=entry.title.strip().replace("\n", " "),
                authors=[a.name for a in entry.authors],
                published=dt.datetime(*entry.published_parsed[:6]),
                citations=cit,
                pdf_url=next((l.href for l in entry.links if l.type == "application/pdf"), None),
            )
    papers = await asyncio.gather(*(handle(e) for e in feed.entries))
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=ARXIV_QUERY_WINDOW_DAYS)
    papers = [p for p in papers if p.published >= cutoff and (p.citations or 0) >= CITATION_THRESHOLD]
    logger.info(f"arXiv qualified: {len(papers)}")
    return papers