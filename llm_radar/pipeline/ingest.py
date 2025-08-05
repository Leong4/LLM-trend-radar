import asyncio, httpx
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from loguru import logger
from ..utils.io import download_pdf
from ..db.models import SessionLocal, PaperORM
from ..db.schema import PaperSchema as Paper
from ..utils.hashing import sha1

async def enrich_and_save(papers: list[Paper]):
    async with httpx.AsyncClient() as client:
        progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), transient=True)
        progress.start()
        task = progress.add_task("Downloading PDFs", total=len(papers))
        sem = asyncio.Semaphore(10)

        async def process(p: Paper):
            async with sem:
                p.pdf_path = await download_pdf(client, p)
                p.status = "full" if p.pdf_path else "gray"
                orm = PaperORM(
                    arxiv_id=p.arxiv_id, title=p.title, authors=", ".join(p.authors),
                    published=p.published, citations=p.citations or 0,
                    pdf_path=p.pdf_path, status=p.status, source_type=p.source_type,
                    fingerprint=sha1(p.title),
                )
                with SessionLocal() as s:
                    try: s.add(orm); s.commit()
                    except: s.rollback()
                progress.advance(task)

        await asyncio.gather(*(process(p) for p in papers))
        progress.stop()