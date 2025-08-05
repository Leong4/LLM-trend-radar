import httpx, pathlib, datetime as dt
from loguru import logger
from ..config import PDF_DIR
from ..db.schema import PaperSchema as Paper

async def download_pdf(client: httpx.AsyncClient, paper: Paper) -> str | None:
    if not paper.pdf_url:
        return None
    fname = PDF_DIR / f"{paper.arxiv_id}.pdf"
    try:
        resp = await client.get(paper.pdf_url, timeout=60)
        resp.raise_for_status()
        fname.write_bytes(resp.content)
        return str(fname)
    except Exception as e:
        logger.error(f"PDF download failed for {paper.arxiv_id}: {e}")
        return None