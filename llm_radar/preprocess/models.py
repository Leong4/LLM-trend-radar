from pydantic import BaseModel
import datetime as dt
from typing import Optional

class DocMeta(BaseModel):
    parse_status: str = "ok"       # 'ok'|'failed'
    parse_error: Optional[str] = None
    process_stage: str = "lite"    # 'lite'|'full'
    lang: Optional[str] = None
    n_pages: Optional[int] = None

class ChunkMeta(BaseModel):
    section_title: Optional[str] = None
    text: str
    n_chars: int
    n_tokens: int
    lang: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    tier: str = "gray"             # 'gray'|'full'|'shadow'
    hash: str