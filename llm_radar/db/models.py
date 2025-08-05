import datetime as dt
from sqlalchemy import Column, DateTime, Integer, String, create_engine
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

engine = create_engine(f"sqlite:///{DATA_DIR/'trend_radar.db'}", future=True)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)