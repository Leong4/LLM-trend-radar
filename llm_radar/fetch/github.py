import datetime as dt, httpx
from loguru import logger
from ..config import GITHUB_TOKEN
from ..db.schema import PaperSchema as Paper

async def fetch_github_batch() -> list[Paper]:
    headers = {
        "Accept": "application/vnd.github+json",
        **({"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}),
    }
    params = {"q": "topic:rag stars:>0", "sort": "stars", "order": "desc", "per_page": 30}
    url = "https://api.github.com/search/repositories"
    async with httpx.AsyncClient(headers=headers, timeout=30) as client:
        resp = await client.get(url, params=params)
        data = resp.json()
    papers = [
        Paper(
            arxiv_id=f"github:{item['id']}",
            title=item["full_name"],
            authors=[item["owner"]["login"]],
            published=dt.datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
            citations=item["stargazers_count"],
            source_type="github",
        )
        for item in data.get("items", [])
    ]
    logger.info(f"GitHub items pulled: {len(papers)}")
    return papers