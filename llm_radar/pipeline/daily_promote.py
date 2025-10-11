from __future__ import annotations
import argparse, datetime as dt, sqlite3, os, json
from pathlib import Path
from loguru import logger
from llm_radar.config import DATA_DIR
from llm_radar.promoter.scoring import blend_by_age  # 仅用于年龄，分数直接取 signals
from llm_radar.utils.conf import load_config

DB_PATH = Path(DATA_DIR) / "trend_radar.db"

# 默认阈值（可后续接 YAML）
TAU_PROB = 0.30
TAU_FULL_YOUNG = 0.65
TAU_FULL_MID   = 0.60
TAU_FULL_OLD   = 0.55
TAU_DEMOTE     = 0.45

# 生命周期控制（试用与回收）
FULL_TRIAL_DAYS = 14     # FULL 试用期，到点复核
PROB_TTL_DAYS   = 30     # PROBATION 超时未改善 → IGNORE

# --- config override (YAML) ---
def apply_config(cfg: dict | None):
    """Override module globals with values from YAML config if provided."""
    if not cfg:
        return
    T = cfg.get("thresholds", {})
    L = cfg.get("lifecycle", {})
    global TAU_PROB, TAU_FULL_YOUNG, TAU_FULL_MID, TAU_FULL_OLD, TAU_DEMOTE
    global FULL_TRIAL_DAYS, PROB_TTL_DAYS
    TAU_PROB = float(T.get("tau_prob", TAU_PROB))
    TAU_FULL_YOUNG = float(T.get("tau_full_young", TAU_FULL_YOUNG))
    TAU_FULL_MID   = float(T.get("tau_full_mid", TAU_FULL_MID))
    TAU_FULL_OLD   = float(T.get("tau_full_old", TAU_FULL_OLD))
    TAU_DEMOTE     = float(T.get("tau_demote", TAU_DEMOTE))
    FULL_TRIAL_DAYS = int(L.get("full_trial_days", FULL_TRIAL_DAYS))
    PROB_TTL_DAYS   = int(L.get("probation_ttl_days", PROB_TTL_DAYS))


def tau_for_age(age_days: int) -> float:
    if age_days <= 14: return TAU_FULL_YOUNG
    if age_days <= 90: return TAU_FULL_MID
    return TAU_FULL_OLD


def fetch_score(conn, item_id: str, day: str):
    cur = conn.cursor()
    row = cur.execute("SELECT score, ucb FROM signals WHERE id=? AND day=?", (item_id, day)).fetchone()
    return (row[0], row[1]) if row else (None, None)


def low_streak_days(conn, item_id: str, day: str, tau_demote: float = TAU_DEMOTE):
    cur = conn.cursor()
    d = dt.date.fromisoformat(day)
    week = (d - dt.timedelta(days=6)).isoformat()
    row = cur.execute(
        "SELECT COUNT(*) FROM signals WHERE id=? AND day>=? AND day<=? AND score<?",
        (item_id, week, day, tau_demote)
    ).fetchone()
    return int(row[0] or 0)


def last_transition_ts(conn, item_id: str, to_state: str):
    """取该 item 最近一次迁入 to_state 的时间；若无则返回 None。"""
    cur = conn.cursor()
    row = cur.execute(
        "SELECT ts FROM transitions WHERE id=? AND to_state=? ORDER BY ts DESC LIMIT 1",
        (item_id, to_state)
    ).fetchone()
    if not row:
        return None
    try:
        return dt.datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
    except Exception:
        return None


def run_daily_promote(day: str):
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    items = cur.execute("SELECT id, source, created_at, state FROM items").fetchall()
    promote_cnt = demote_cnt = stay_cnt = 0

    for (item_id, source, created_at, state) in items:
        # 年龄
        try:
            age_days = (dt.date.fromisoformat(day) - dt.date.fromisoformat(str(created_at)[:10])).days
        except Exception:
            age_days = 0

        S, U = fetch_score(conn, item_id, day)
        if S is None:  # 今天没算到分
            continue

        next_state, reason = state, "stay"
        if state == "FULL":
            # 1) 连续 7 天低分降级
            streak = low_streak_days(conn, item_id, day, TAU_DEMOTE)
            if streak >= 7 and S < TAU_DEMOTE:
                next_state, reason = "PROBATION", "demote_low_trend"
                demote_cnt += 1
            else:
                # 2) FULL 试用到点复核（14 天）
                last_full = last_transition_ts(conn, item_id, "FULL")
                if last_full is not None:
                    days_in_full = (dt.date.fromisoformat(day) - last_full.date()).days
                else:
                    # 若无迁移记录，尽量用 created_at 粗估；否则视为未到试用点
                    try:
                        created_date = dt.date.fromisoformat(str(created_at)[:10])
                        days_in_full = (dt.date.fromisoformat(day) - created_date).days
                    except Exception:
                        days_in_full = 0
                if days_in_full >= FULL_TRIAL_DAYS and S < TAU_DEMOTE:
                    next_state, reason = "PROBATION", "trial_review_demote"
                    demote_cnt += 1
                else:
                    stay_cnt += 1
        else:
            # PROBATION / IGNORE
            tau = tau_for_age(age_days)
            # 保持当前探索强度：年轻/中年段允许 UCB 晋级；老样本仅看 S
            pass_ucb = (age_days <= 14 and U is not None and U >= 0.70) or \
                       (14 < age_days <= 90 and U is not None and U >= 0.68)

            # 先判晋级，再看是否 TTL 回收
            if S >= tau or pass_ucb:
                next_state, reason = "FULL", "promote_score_or_ucb"
                promote_cnt += 1
            else:
                # PROBATION TTL：30 天无改善 → IGNORE
                last_prob = last_transition_ts(conn, item_id, "PROBATION")
                if last_prob is not None:
                    days_in_prob = (dt.date.fromisoformat(day) - last_prob.date()).days
                else:
                    try:
                        created_date = dt.date.fromisoformat(str(created_at)[:10])
                        days_in_prob = (dt.date.fromisoformat(day) - created_date).days
                    except Exception:
                        days_in_prob = 0
                if state == "PROBATION" and days_in_prob >= PROB_TTL_DAYS:
                    next_state, reason = "IGNORE", "probation_ttl"
                    demote_cnt += 1
                elif S >= TAU_PROB:
                    next_state, reason = "PROBATION", "stay_probation"
                    stay_cnt += 1
                else:
                    next_state, reason = "PROBATION", "new_probation"
                    stay_cnt += 1

        if next_state != state:
            cur.execute("UPDATE items SET state=?, last_score=?, last_ucb=? WHERE id=?",
                        (next_state, S, U, item_id))
            cur.execute(
                "INSERT INTO transitions (id, ts, from_state, to_state, reason, details) VALUES (?, ?, ?, ?, ?, ?)",
                (item_id, dt.datetime.utcnow().isoformat(), state, next_state, reason, json.dumps({"score": S, "ucb": U, "age_days": age_days}))
            )
        else:
            # 也同步 last_score/ucb，便于观测
            cur.execute("UPDATE items SET last_score=?, last_ucb=? WHERE id=?", (S, U, item_id))

    # === 指标导出 & 审计导出 ===
    # 全量指标
    row = cur.execute("SELECT COUNT(*) FROM items WHERE state='FULL'").fetchone()
    full_cnt = int(row[0] or 0)
    row = cur.execute("SELECT AVG(last_score), AVG(last_ucb) FROM items").fetchone()
    avg_s = round(row[0] or 0, 4)
    avg_u = round(row[1] or 0, 4)

    exp_dir = Path("exports"); exp_dir.mkdir(exist_ok=True)

    # 1) metrics_daily.csv（追加写）
    m_path = exp_dir / "metrics_daily.csv"
    write_header = not m_path.exists()
    with m_path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write("date,full_count,promotions,demotions,avg_score,avg_ucb\n")
        f.write(f"{day},{full_cnt},{promote_cnt},{demote_cnt},{avg_s},{avg_u}\n")

    # 2) 当日迁移明细（覆盖写）
    a_path = exp_dir / f"audit_transitions_{day}.csv"
    rows = cur.execute(
        "SELECT id, ts, from_state, to_state, reason, details FROM transitions WHERE date(ts)=?",
        (day,)
    ).fetchall()
    with a_path.open("w", encoding="utf-8") as f:
        f.write("id,ts,from,to,reason,details\n")
        for r in rows:
            did = str(r[0]).replace(",", " ")
            ts  = str(r[1])
            frm = str(r[2])
            to  = str(r[3])
            rsn = str(r[4])
            dts = str(r[5]).replace("\n", " ").replace(",", ";")
            f.write(f"{did},{ts},{frm},{to},{rsn},{dts}\n")

    conn.commit()
    conn.close()
    logger.info(f"promote done → FULL:+{promote_cnt}  demote:{demote_cnt}  stay:{stay_cnt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--config", type=str, default=None, help="path to config/promoter.yaml")
    args = ap.parse_args()

    # load YAML config (if available) and override globals
    try:
        cfg = load_config(args.config)
    except Exception:
        cfg = None
    apply_config(cfg)

    run_daily_promote(args.date)


if __name__ == "__main__":
    main()