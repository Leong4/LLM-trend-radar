#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radar Weekly Report (Markdown)
- Summarize promotions/demotions in last N days
- List "New but Promising" from PROBATION by high UCB
- Basic activity metrics
"""
import argparse, sqlite3, datetime as dt, os, json
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser("Radar weekly reporter")
    ap.add_argument("--db", default="data/trend_radar.db")
    ap.add_argument("--days", type=int, default=7, help="lookback days")
    ap.add_argument("--out", default=None, help="output md file, default: reports/weekly_YYYYWW.md")
    ap.add_argument("--probation_ucb_topk", type=int, default=8)
    return ap.parse_args()

def yyyww(d: dt.date) -> str:
    iso = d.isocalendar()
    return f"{iso.year}-{iso.week:02d}"

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def fetch_one(con, q, params=()):
    cur = con.execute(q, params)
    row = cur.fetchone()
    return row if row else (0,)

def fetch_all(con, q, params=()):
    cur = con.execute(q, params)
    return cur.fetchall()

def main():
    args = parse_args()
    con = sqlite3.connect(args.db)
    today = dt.datetime.utcnow().date()
    since = today - dt.timedelta(days=args.days)
    week_tag = yyyww(today)

    # promotions / demotions
    trans = fetch_all(con, """
    SELECT ts, from_state, to_state, reason, id
    FROM transitions
    WHERE ts >= ?
    ORDER BY ts DESC
    """, (since.isoformat(),))

    # join titles
    doc_map = {row[0]: row[1:] for row in fetch_all(con, "SELECT id, title, source_type FROM doc", ())}
    def meta(doc_id):
        t = doc_map.get(doc_id, ("", "")) 
        return t[0] or f"Doc {doc_id}", t[1] or "?"

    promotions = []
    demotions  = []
    trial_demos = 0
    for ts, st, nxt, reason, did in trans:
        title, src = meta(did)
        item = dict(ts=ts, id=did, title=title, source=src, reason=reason or "", from_state=st, to_state=nxt)
        # promotion: moved into FULL
        if st != "FULL" and nxt == "FULL":
            promotions.append(item)
        # demotion: left FULL (to PROBATION/IGNORE/etc.)
        elif st == "FULL" and nxt != "FULL":
            demotions.append(item)
            if (reason or "").strip() == "trial_review_demote":
                trial_demos += 1

    # probation high-UCB candidates (latest per id)
    # prefer latest signals for each id
    prob_top = fetch_all(con, """
    WITH latest AS (
      SELECT s.id, MAX(s.day) AS last_day
      FROM signals s
      JOIN items i ON i.id = s.id
      WHERE i.state='PROBATION'
      GROUP BY s.id
    )
    SELECT s.id, s.ucb, s.score, i.state,
           COALESCE(d.title, '') AS title,
           COALESCE(d.source_type, i.source, '?') AS source
    FROM signals s
    JOIN latest L ON L.id=s.id AND L.last_day=s.day
    LEFT JOIN items i ON i.id=s.id
    LEFT JOIN doc d ON d.id=s.id
    ORDER BY s.ucb DESC, s.score DESC
    LIMIT ?
    """, (args.probation_ucb_topk,))

    # activity metrics
    # chunks written in window (fallback: doc.created_at)
    chunks = fetch_one(con, """
    SELECT COUNT(*)
    FROM doc_chunk c
    LEFT JOIN doc d ON d.id = c.doc_id
    WHERE COALESCE(c.created_at, d.created_at, (SELECT MIN(ts) FROM transitions)) >= ?
    """, (since.isoformat(),))
    full_now = fetch_one(con, "SELECT COUNT(*) FROM items WHERE state='FULL'")
    prob_now = fetch_one(con, "SELECT COUNT(*) FROM items WHERE state='PROBATION'")

    # render markdown
    lines = []
    lines.append(f"# Radar Weekly — {week_tag}\n")
    lines.append("## New but Promising (UCB Explore)\n")
    lines.append("| ID | Title | Source | UCB | Score S | State |")
    lines.append("|---|---|---|---:|---:|---|")

    def pretty_title(rid: str, title: str) -> str:
        if title:
            return title
        if rid.startswith("gh:"):
            pos = rid.rfind("/")
            return rid[pos+1:] if pos > 0 else rid
        return rid

    for rid, ucb, sc, st, title, src in prob_top:
        pt = pretty_title(rid, title)
        lines.append(f"| {rid} | {pt} | {src or '?'} | {ucb:.3f} | {sc:.3f} | {st} |")

    lines.append("\n## Promotions to FULL\n")
    if promotions:
        lines.append("| ts | ID | Title | Source | Reason |")
        lines.append("|---|---|---|---|---|")
        for x in promotions:
            lines.append(f"| {x['ts']} | {x['id']} | {x['title']} | {x['source']} | {x['reason']} |")
    else:
        lines.append("_No promotions in window._")

    lines.append("\n## Demotions / TTL Reclaims\n")
    if demotions:
        lines.append("| ts | ID | Title | Source | Reason |")
        lines.append("|---|---|---|---|---|")
        for x in demotions:
            lines.append(f"| {x['ts']} | {x['id']} | {x['title']} | {x['source']} | {x['reason']} |")
    else:
        lines.append("_No demotions in window._")

    lines.append("\n## Quick Metrics\n")
    lines.append(f"- FULL now: **{full_now[0]}**")
    lines.append(f"- PROBATION now: **{prob_now[0]}**")
    lines.append(f"- Chunks written (last {args.days}d): **{chunks[0]}**")
    lines.append(f"- Demotions due to `trial_review_demote` (last {args.days}d): **{trial_demos}**")
    lines.append("")

    md = "\n".join(lines)

    out = args.out or f"reports/weekly_{week_tag}.md"
    outp = Path(out)
    ensure_dir(outp)
    outp.write_text(md, encoding="utf-8")
    print(f"Wrote {outp} ({len(lines)} lines)")

if __name__ == "__main__":
    main()