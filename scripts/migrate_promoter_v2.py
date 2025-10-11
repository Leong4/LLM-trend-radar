#!/usr/bin/env python3
import argparse, json, datetime as dt, sqlite3, os, re
from pathlib import Path

# 读配置中的 DB 路径
from llm_radar.config import DATA_DIR

SQL_PATH = Path(__file__).resolve().parents[1] / "llm_radar" / "db" / "migrations" / "0003_promoter_v2.sql"
DB_PATH = Path(DATA_DIR) / "trend_radar.db"

def load_sql(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def to_item_id(source: str, title: str, authors: str, arxiv_id: str | None) -> str:
    if source == "arxiv":
        return f"ax:{arxiv_id}"
    # github: 允许 title 是 'owner/repo' 或只有 'repo'，authors 里逗号分隔的第一个作为 owner 兜底
    repo = (title or "").strip()
    if "/" in repo:
        owner_repo = repo
    else:
        owner = (authors or "").split(",")[0].strip() or "unknown"
        owner_repo = f"{owner}/{repo or 'unknown'}"
    owner_repo = re.sub(r"\s+", "-", owner_repo)
    return f"gh:{owner_repo.lower()}"

def seed_from_paper(conn: sqlite3.Connection, dry_run: bool):
    cur = conn.cursor()
    # 选择现有 paper 用于灌入
    rows = cur.execute("""
        SELECT arxiv_id, title, authors, published, citations, pdf_path, status, source_type, created_at
          FROM paper
    """).fetchall()

    inserted = 0
    for (arxiv_id, title, authors, published, citations, pdf_path, status, source_type, created_at) in rows:
        source = source_type or ("arxiv" if arxiv_id else "github")
        item_id = to_item_id(source, title or "", authors or "", arxiv_id)
        # 初始状态映射：full -> FULL，其它 -> PROBATION（灰名单先进入观察期）
        state = "FULL" if (status or "").lower() == "full" else "PROBATION"
        created = None
        if published:
            # published 是 ISO 字符串或 datetime 序列化；尝试解析
            try:
                created = dt.datetime.fromisoformat(str(published).replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                created = None
        if created is None:
            try:
                created = dt.datetime.fromisoformat(str(created_at).replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                created = dt.datetime.utcnow()

        meta = {
            "title": title, "authors": authors, "arxiv_id": arxiv_id,
            "citations": citations, "pdf_path": pdf_path,
            "source_type": source_type, "paper_status": status,
        }
        if not dry_run:
            cur.execute("""
                INSERT INTO items (id, source, created_at, state, last_score, last_ucb, obs_days, cluster_id, meta_json)
                VALUES (?, ?, ?, ?, 0, 0, 0, NULL, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source=excluded.source,
                    created_at=excluded.created_at
            """, (item_id, source, created, state, json.dumps(meta)))
            inserted += 1

    if not dry_run:
        conn.commit()
    return inserted, len(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只创建表和统计，不落库")
    ap.add_argument("--no-seed", action="store_true", help="只建表，不从 paper 灌入 items")
    args = ap.parse_args()

    print(f"[info] DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    sql = load_sql(SQL_PATH)
    conn.executescript(sql)
    print("[ok] tables ensured.")

    if not args.no_seed:
        ins, total = seed_from_paper(conn, dry_run=args.dry_run)
        print(f"[ok] seeded items from paper: inserted/updated={ins} of total={total}")
    conn.close()

if __name__ == "__main__":
    main()