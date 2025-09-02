import hashlib
import httpx
import feedparser
from loguru import logger

from ..db.models import PaperORM
from .models import DocMeta, ChunkMeta
from .clean import normalize_text, strip_math, strip_code_blocks_md
from .chunk import window_chunks
from .markdown import first_paragraphs

def _hash(section: str, text: str) -> str:
    h = hashlib.sha1()
    h.update(((section or "") + (text or "")[:80]).encode("utf-8"))
    return h.hexdigest()

async def _fetch_arxiv_abstract(arxiv_id: str) -> str | None:
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    async with httpx.AsyncClient() as cli:
        try:
            xml = (await cli.get(url, timeout=20)).text
            feed = feedparser.parse(xml)
            if not feed.entries:
                return None
            return feed.entries[0].get("summary", None)
        except Exception as e:
            logger.warning(f"arXiv abstract fetch failed: {e}")
            return None

async def preprocess_arxiv_lite(p: PaperORM):
    """
    输入: PaperORM (arxiv, gray)
    输出: DocMeta, [ChunkMeta]  — 仅 1 个chunk: 标题 + 摘要
    """
    doc = DocMeta(process_stage="lite", lang=None)
    abstract = await _fetch_arxiv_abstract(p.arxiv_id)
    text = f"{p.title}\n\n{abstract or ''}"
    text = strip_math(normalize_text(text))
    chunks_text = window_chunks(text, max_tokens=400, overlap=60, min_tokens=1)
    chunks_text = chunks_text[:1]  # lite 只保留首块

    chunks = []
    for i, txt in enumerate(chunks_text):
        chunks.append(ChunkMeta(
            section_title="Abstract",
            text=txt,
            n_chars=len(txt),
            n_tokens=max(1, len(txt.split())),
            lang=None,
            page_start=None, page_end=None,
            tier="gray",
            hash=_hash("Abstract", txt),
        ))
    return doc, chunks

async def _fetch_readme(owner: str, repo: str, token: str | None):
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = {"Accept": "application/vnd.github.raw"}
    if token:
        headers["Authorization"] = f"token {token}"
    async with httpx.AsyncClient() as cli:
        r = await cli.get(url, headers=headers, timeout=20)
        if r.status_code >= 400:
            return None
        return r.text

async def preprocess_github_lite(p: PaperORM, token: str | None = None):
    """
    输入: PaperORM (github, gray)
    输出: DocMeta, [ChunkMeta] — README 前两段或前2KB，1 个chunk
    """
    doc = DocMeta(process_stage="lite", lang=None)
    owner = (p.authors or "").split(",")[0].strip()
    # title 可能是 'repo' 或 'owner/repo'
    if "/" in p.title:
        owner_repo = p.title
    else:
        owner_repo = f"{owner}/{p.title}"
    try:
        owner, repo = owner_repo.split("/", 1)
    except ValueError:
        return DocMeta(parse_status="failed", parse_error="bad_repo", process_stage="lite"), []

    md = await _fetch_readme(owner, repo, token)
    if not md:
        return DocMeta(parse_status="failed", parse_error="no_readme", process_stage="lite"), []

    text = first_paragraphs(md, max_bytes=2048, max_pars=2)
    text = strip_code_blocks_md(normalize_text(text))
    chunks_text = window_chunks(text, max_tokens=400, overlap=60, min_tokens=1)
    chunks_text = chunks_text[:1]

    chunks = []
    for i, txt in enumerate(chunks_text):
        chunks.append(ChunkMeta(
            section_title="README",
            text=txt,
            n_chars=len(txt),
            n_tokens=max(1, len(txt.split())),
            lang=None,
            page_start=None, page_end=None,
            tier="gray",
            hash=_hash("README", txt),
        ))
    return doc, chunks