import datetime as dt
from typing import List
from pydantic import BaseModel, Field

class PaperSchema(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str]
    published: dt.datetime
    citations: int | None = None
    pdf_url: str | None = None
    pdf_path: str | None = None
    status: str = "gray"
    last_checked: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    gray_reason: str | None = None
    source_type: str = "arxiv"
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    lang: str | None = None

    class Config:
        arbitrary_types_allowed = True