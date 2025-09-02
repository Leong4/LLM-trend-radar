import datetime as dt
from sqlalchemy import Column, DateTime, Integer, String, create_engine, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from ..config import DATA_DIR


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
    status = Column(String, default="gray")
    last_checked = Column(DateTime, default=dt.datetime.utcnow)
    gray_reason = Column(String, nullable=True)
    source_type = Column(String, default="arxiv")
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    lang = Column(String, nullable=True)
    fingerprint = Column(String, unique=True)

# 新增文档表（doc）
class DocORM(Base):
    __tablename__ = "doc"
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey("paper.id"), index=True)
    source_type = Column(String, index=True)  # 'arxiv' | 'github'
    title = Column(String)
    authors = Column(String)
    published = Column(DateTime)
    lang = Column(String, nullable=True)
    n_pages = Column(Integer, nullable=True)
    parse_status = Column(String, default="ok")     # 'ok'|'failed'
    parse_error = Column(String, nullable=True)
    process_stage = Column(String)                  # 'lite'|'full'
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    updated_at = Column(DateTime, default=dt.datetime.utcnow)

# 新增切块表（doc_chunk）
class DocChunkORM(Base):
    __tablename__ = "doc_chunk"
    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer, ForeignKey("doc.id"), index=True)
    section_title = Column(String, nullable=True)
    chunk_index = Column(Integer)   # 文档内顺序
    text = Column(String)
    n_chars = Column(Integer)
    n_tokens = Column(Integer)
    lang = Column(String, nullable=True)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    hash = Column(String, unique=True)  # sha1(section + text[:80])
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    tier = Column(String)               # 'gray'|'full'|'shadow'

engine = create_engine(f"sqlite:///{DATA_DIR/'trend_radar.db'}", future=True)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)