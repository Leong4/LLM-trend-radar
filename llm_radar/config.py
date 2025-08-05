import os, pathlib, datetime as dt

ROOT_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
PDF_DIR  = DATA_DIR / "papers_pdf"
LOG_DIR  = ROOT_DIR / "logs"

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