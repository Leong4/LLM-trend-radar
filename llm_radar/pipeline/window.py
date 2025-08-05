from ..config import STATS_FILE, ARXIV_QUERY_WINDOW_DAYS
import datetime as dt, json, statistics

def adjust_arxiv_window(new_count: int) -> int:
    """Adjust ARXIV_QUERY_WINDOW_DAYS based on last fetch volume."""
    history = []
    if STATS_FILE.exists():
        history = json.loads(STATS_FILE.read_text())
    history.append({"ts": dt.datetime.utcnow().isoformat(), "new": new_count})
    history = history[-7:]  # keep last 7 records
    STATS_FILE.write_text(json.dumps(history, indent=2))

    avg = statistics.mean(h["new"] for h in history)
    wnd = ARXIV_QUERY_WINDOW_DAYS
    if avg < 20:
        wnd = min(wnd + 2, 14)
    elif avg > 100:
        wnd = max(wnd - 1, 3)
    return wnd