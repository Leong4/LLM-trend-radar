import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent           # …/project1/llm_radar
PROJECT_ROOT = PKG_DIR.parent                        # …/project1

DATA_DIR = Path(os.getenv("TREND_RADAR_DATA_DIR", PROJECT_ROOT / "data"))
PDF_DIR  = DATA_DIR / "papers_pdf"
LOG_DIR  = PROJECT_ROOT / "logs"

for d in (DATA_DIR, PDF_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Global params
STATS_FILE = DATA_DIR / "stats.json"
ARXIV_CATEGORY = "cs.CL"
ARXIV_QUERY_WINDOW_DAYS = 7
CITATION_THRESHOLD = 0
CONCURRENT_REQUESTS = 10

SEM_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_KEY")
GITHUB_TOKEN        = os.getenv("GITHUB_TOKEN")