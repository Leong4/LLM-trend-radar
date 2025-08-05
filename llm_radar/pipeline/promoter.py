import datetime as dt
import httpx
from sqlalchemy import select
from loguru import logger
from ..db.models import SessionLocal, PaperORM
from ..config import GITHUB_TOKEN
from ..fetch.arxiv import fetch_citations

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