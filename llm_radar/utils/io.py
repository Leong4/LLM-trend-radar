import httpx, pathlib, datetime as dt
from loguru import logger
from ..config import PDF_DIR
from ..db.schema import PaperSchema as Paper


async def download_pdf(client: httpx.AsyncClient, paper: Paper) -> str | None:
    if not paper.pdf_url:
        return None

    url = paper.pdf_url

    # --- arXiv 链接规范化：强制 https、确保以 .pdf 结尾 ---
    if "arxiv.org" in url:
        if url.startswith("http://"):
            url = "https://" + url[len("http://"):]
        if "/pdf/" in url and not url.endswith(".pdf"):
            url = url + ".pdf"

    fname = PDF_DIR / f"{paper.arxiv_id}.pdf"

    try:
        # 关键：跟随重定向
        resp = await client.get(url, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        fname.write_bytes(resp.content)
        return str(fname)
    except httpx.HTTPStatusError as e:
        logger.error(
            f"PDF download failed for {paper.arxiv_id}: "
            f"{e.response.status_code} {e.response.reason_phrase} for url {url}"
        )
        return None
    except Exception as e:
        logger.error(f"PDF download failed for {paper.arxiv_id}: {e}")
        return None