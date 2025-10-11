# llm_radar/pipeline/daily_features.py
from __future__ import annotations
import argparse, datetime as dt, json, os, re, sqlite3, httpx, html
from pathlib import Path
from loguru import logger

from llm_radar.config import DATA_DIR
from llm_radar.promoter.scoring import normalize, blend_by_age, ucb
from llm_radar.utils.conf import load_config

DB_PATH = Path(DATA_DIR) / "trend_radar.db"
TODAY = dt.date.today().isoformat()

# ---- config-overridable defaults ----
READMELEN_MAX = 8000
DELTA_STAR_WIN = 7      # days for Δlevel (stars/citations)
COMMIT_WIN = 14         # days for commit window
ISSUE_AUTHOR_WIN = 14   # days for uniq issue/PR authors window
UCB_C = 1.0             # UCB coefficient

# 归一化上限与兜底
STAR_CAP = 50          # GitHub: 7d star 增量的上限归一化
CIT_CAP  = 10          # arXiv: 7d 引用增量的上限归一化
ACTOR_CAP = 10         # GitHub: 14d 参与者上限归一化
HALF_LIFE_DEFAULT_T = 30  # 系统运行天数兜底（用于 UCB）

# ---- 轻量配置（可被 config/promoter.yaml 覆盖） ----
ARXIV_CODE_LINK_PAT = re.compile(r"(github\.com|gitlab\.com|huggingface\.co|paperswithcode\.com|bitbucket\.org)", re.I)
ARXIV_TOPIC_KEYWORDS = [
    # LLM / RAG 热点关键词（可扩展）
    "rag", "retrieval", "retriever", "vector", "embedding", "rerank",
    "llm", "large language model", "tool use", "agent", "memory",
    "distillation", "inference", "prompt", "context window", "chunk",
]
LAB_WHITELIST = ["DeepMind", "FAIR", "Meta AI", "Google", "MSR", "OpenAI", "Tsinghua", "CMU", "Stanford", "MIT"]
ORG_WHITELIST = ["openai", "deepmind", "google", "microsoft", "meta", "huggingface"]

def apply_config(cfg: dict | None):
    """Override module-level defaults from YAML config if provided."""
    if not cfg:
        return
    try:
        GH = (cfg or {}).get("features", {}).get("github", {})
        WL = (cfg or {}).get("whitelists", {})
        KW = (cfg or {}).get("keywords", {})
        U  = (cfg or {}).get("ucb", {})

        global READMELEN_MAX, STAR_CAP, ACTOR_CAP, DELTA_STAR_WIN, COMMIT_WIN, ISSUE_AUTHOR_WIN, UCB_C
        READMELEN_MAX   = int(GH.get("readme_len_max", READMELEN_MAX))
        STAR_CAP        = int(GH.get("star_cap", STAR_CAP))
        ACTOR_CAP       = int(GH.get("actor_cap", ACTOR_CAP))
        DELTA_STAR_WIN  = int(GH.get("delta_star_window_days", DELTA_STAR_WIN))
        COMMIT_WIN      = int(GH.get("commit_window_days", COMMIT_WIN))
        ISSUE_AUTHOR_WIN= int(GH.get("uniq_issue_authors_window_days", ISSUE_AUTHOR_WIN))
        UCB_C           = float(U.get("c", UCB_C))

        global LAB_WHITELIST, ORG_WHITELIST, ARXIV_TOPIC_KEYWORDS
        labs = WL.get("labs")
        if isinstance(labs, list) and labs:
            LAB_WHITELIST = labs
        orgs = WL.get("orgs")
        if isinstance(orgs, list) and orgs:
            ORG_WHITELIST = [o.lower() for o in orgs]
        kws = KW.get("llm_rag")
        if isinstance(kws, list) and kws:
            ARXIV_TOPIC_KEYWORDS = kws
    except Exception as e:
        logger.warning(f"apply_config failed: {e}")

# ---------------- util helpers ----------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def has_usage_section(md: str) -> bool:
    # 简单检测 README 中是否包含 Usage/Install/Example 等段落
    return bool(re.search(r"(usage|install|examples?)", md, re.IGNORECASE))

# --------- GitHub helpers ----------

def gh_headers(token: str | None):
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

async def gh_raw(cli: httpx.AsyncClient, path: str, token: str | None):
    url = f"https://api.github.com{path}"
    return await cli.get(url, headers=gh_headers(token), timeout=30)

async def gh_get_repo(cli: httpx.AsyncClient, owner_repo: str, token: str | None):
    r = await gh_raw(cli, f"/repos/{owner_repo}", token)
    r.raise_for_status()
    return r.json()

async def gh_get_readme(cli: httpx.AsyncClient, owner_repo: str, token: str | None) -> str:
    # 直接拉 raw 内容，兼容默认分支名
    url = f"https://raw.githubusercontent.com/{owner_repo}/HEAD/README.md"
    r = await cli.get(url, timeout=30)
    return r.text if r.status_code == 200 else ""

async def gh_path_exists(cli: httpx.AsyncClient, owner_repo: str, path: str, token: str | None) -> bool:
    # /repos/{owner_repo}/contents/{path}
    r = await gh_raw(cli, f"/repos/{owner_repo}/contents/{path}", token)
    return r.status_code == 200

async def gh_commit_count_window(cli: httpx.AsyncClient, owner_repo: str, token: str | None, days: int) -> int:
    since = (dt.datetime.utcnow() - dt.timedelta(days=days)).isoformat() + "Z"
    r = await gh_raw(cli, f"/repos/{owner_repo}/commits?since={since}", token)
    if r.status_code == 200:
        try:
            return len(r.json())
        except Exception:
            return 0
    return 0

async def gh_uniq_issue_authors_window(cli: httpx.AsyncClient, owner_repo: str, token: str | None, days: int) -> int:
    """拉取近 N 天内更新的 issues（包含 PR），统计去重作者数。"""
    since = (dt.datetime.utcnow() - dt.timedelta(days=days)).isoformat() + "Z"
    r = await gh_raw(cli, f"/repos/{owner_repo}/issues?since={since}&state=all&per_page=100", token)
    uniq = set()
    if r.status_code == 200:
        try:
            for it in r.json():
                user = (it or {}).get("user") or {}
                if user.get("login"):
                    uniq.add(user["login"].lower())
                for a in (it or {}).get("assignees") or []:
                    if a.get("login"):
                        uniq.add(a["login"].lower())
        except Exception:
            pass
    return len(uniq)

# --------- arXiv helpers ----------

def get_citations_from_meta(meta_json: str | None) -> int:
    try:
        if not meta_json:
            return 0
        meta = json.loads(meta_json)
        return int(meta.get("citations") or 0)
    except Exception:
        return 0

async def fetch_arxiv_abstract(cli: httpx.AsyncClient, ax_id: str) -> str:
    """通过 arXiv API 拉摘要；失败返回空串。"""
    try:
        url = f"https://export.arxiv.org/api/query?id_list={ax_id}"
        r = await cli.get(url, timeout=30)
        if r.status_code != 200:
            return ""
        # 粗暴解析 <summary>...</summary>
        m = re.search(r"<summary>(.*?)</summary>", r.text, re.S|re.I)
        if not m:
            return ""
        return html.unescape(m.group(1)).strip()
    except Exception:
        return ""

# ---- arXiv prior: 轻规则 ----

def has_code_or_data_link(text: str) -> bool:
    return bool(ARXIV_CODE_LINK_PAT.search(text or ""))

def topic_keywords_score(text: str) -> float:
    if not text:
        return 0.0
    lower = text.lower()
    hits = sum(1 for kw in ARXIV_TOPIC_KEYWORDS if kw in lower)
    return clamp01(hits / max(3, len(ARXIV_TOPIC_KEYWORDS)//2))  # 粗归一

def lab_whitelist_bonus(meta_json: str | None) -> float:
    try:
        meta = json.loads(meta_json or '{}')
        title = (meta.get('title') or '').lower()
        authors = (meta.get('authors') or '').lower()
        for lab in LAB_WHITELIST:
            l = lab.lower()
            if l in title or l in authors:
                return 0.05
    except Exception:
        pass
    return 0.0

# --------- DB helpers ----------

def ensure_signals_level_column(cur: sqlite3.Cursor) -> None:
    """确保 signals 表存在 level 列（原始量级：stars/citations）。"""
    cols = [row[1] for row in cur.execute("PRAGMA table_info(signals)").fetchall()]
    if "level" not in cols:
        cur.execute("ALTER TABLE signals ADD COLUMN level REAL DEFAULT 0")


def delta_window(cur: sqlite3.Cursor, item_id: str, date_str: str, today_level: float, window_days: int) -> float:
    """返回 N 天增量（若缺历史数据则按 0 处理）。"""
    mod = f"-{int(window_days)} day"
    row = cur.execute(
        "SELECT level FROM signals WHERE id=? AND day=date(?, ?)",
        (item_id, date_str, mod)
    ).fetchone()
    prev = float(row[0]) if row and row[0] is not None else 0.0
    return max(today_level - prev, 0.0)

# --------- core job ----------
async def run_daily(date_str: str, github_token: str | None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 第一次运行时确保 signals.level 存在
    ensure_signals_level_column(cur)

    # 读取 items
    rows = cur.execute(
        "SELECT id, source, created_at, state, last_score, last_ucb, obs_days, meta_json FROM items"
    ).fetchall()
    logger.info(f"items loaded: {len(rows)}")

    # 估算系统运行天数 T（用于 UCB）
    min_created = cur.execute("SELECT MIN(created_at) FROM items").fetchone()[0]
    try:
        first = dt.date.fromisoformat(str(min_created)[:10])
        today = dt.date.fromisoformat(date_str)
        sys_days = (today - first).days + 1
    except Exception:
        sys_days = HALF_LIFE_DEFAULT_T

    async with httpx.AsyncClient(follow_redirects=True) as cli:
        for (item_id, source, created_at, state, last_score, last_ucb, obs_days, meta_json) in rows:
            # 年龄
            try:
                age_days = (dt.date.fromisoformat(date_str) - dt.date.fromisoformat(str(created_at)[:10])).days
            except Exception:
                age_days = 0

            prior = 0.1
            semantic = 0.1  
            reputation = 0.1
            velocity = 0.0
            level = 0.0

            if source == "github" and item_id.startswith("gh:"):
                owner_repo = item_id.split("gh:", 1)[1]
                stars = 0
                readme = ""
                has_tests = False
                has_ci = False
                commit_14d = 0
                uniq_authors_14d = 0
                try:
                    repo = await gh_get_repo(cli, owner_repo, github_token)
                    stars = int(repo.get("stargazers_count", 0))
                except Exception as e:
                    logger.warning(f"gh repo fail {owner_repo}: {e}")

                try:
                    readme = await gh_get_readme(cli, owner_repo, github_token)
                except Exception:
                    readme = ""

                try:
                    has_tests = await gh_path_exists(cli, owner_repo, "tests", github_token)
                except Exception:
                    has_tests = False
                try:
                    has_ci = await gh_path_exists(cli, owner_repo, ".github/workflows", github_token)
                except Exception:
                    has_ci = False
                try:
                    commit_14d = await gh_commit_count_window(cli, owner_repo, github_token, COMMIT_WIN)
                except Exception:
                    commit_14d = 0
                try:
                    uniq_authors_14d = await gh_uniq_issue_authors_window(cli, owner_repo, github_token, ISSUE_AUTHOR_WIN)
                except Exception:
                    uniq_authors_14d = 0

                level = float(stars)                               # 当日原始量级
                vel_raw = delta_window(cur, item_id, date_str, level, DELTA_STAR_WIN)   # 7 天增量（stars）
                velocity = max(
                    normalize(vel_raw, 0, STAR_CAP),                 # Δstars_7d
                    normalize(uniq_authors_14d, 0, ACTOR_CAP),       # 14d 去重参与者
                    normalize(commit_14d, 0, 30)                     # 方案A：把 14d 提交数也纳入趋势
                )

                # 轻量先验：README 长度 + Usage 段 + tests/CI + 近14天提交 + 组织白名单
                prior = 0.0
                prior = max(prior, normalize(len(readme), 0, READMELEN_MAX))
                prior = max(prior, 0.6 if has_usage_section(readme) else 0.0)
                prior = max(prior, 0.4 if (has_tests or has_ci) else 0.0)
                prior = max(prior, normalize(commit_14d, 0, 30))
                # 简单 reputation：组织白名单小加分
                try:
                    owner = owner_repo.split("/", 1)[0].lower()
                    if owner in ORG_WHITELIST:
                        reputation = max(reputation, 0.2)
                except Exception:
                    pass

            elif source == "arxiv" and item_id.startswith("ax:"):
                # 引用增量作为速度
                cits = get_citations_from_meta(meta_json)
                level = float(cits)
                vel_raw = delta_window(cur, item_id, date_str, level, DELTA_STAR_WIN)
                velocity = normalize(vel_raw, 0, CIT_CAP)

                # ---- prior & semantic & reputation（轻量规则） ----
                ax_id = item_id.split("ax:", 1)[1]
                abstract = await fetch_arxiv_abstract(cli, ax_id)
                text_for_topics = abstract
                has_link = has_code_or_data_link(abstract)
                topic_sim = topic_keywords_score(text_for_topics)
                prior  = clamp01(0.1 + (0.2 if has_link else 0.0) + topic_sim*0.3 + lab_whitelist_bonus(meta_json))
                semantic = clamp01(topic_sim)  # 暂以关键词命中率近似语义热度
                reputation = max(reputation, lab_whitelist_bonus(meta_json))

            # 计算 S/U（保持探索强度 c=1.0）
            S = blend_by_age(prior, velocity, semantic, reputation, age_days)
            U = ucb(m=velocity, n=max(1, obs_days), T=max(1, sys_days), c=UCB_C)

            # signals upsert（包含 level）
            cur.execute(
                """
                INSERT INTO signals (id, day, prior, velocity, semantic, reputation, score, ucb, level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id, day) DO UPDATE SET
                    prior=excluded.prior,
                    velocity=excluded.velocity,
                    semantic=excluded.semantic,
                    reputation=excluded.reputation,
                    score=excluded.score,
                    ucb=excluded.ucb,
                    level=excluded.level
                """,
                (item_id, date_str, prior, velocity, semantic, reputation, S, U, level)
            )

            # items last_* & obs_days
            cur.execute(
                """
                UPDATE items SET last_score=?, last_ucb=?, obs_days=COALESCE(obs_days,0)+1
                WHERE id=?
                """,
                (S, U, item_id)
            )

    conn.commit()
    conn.close()
    logger.info("daily_features done.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=TODAY, help="YYYY-MM-DD")
    ap.add_argument("--github-token", default=os.getenv("GITHUB_TOKEN"))
    ap.add_argument("--config", type=str, default=None, help="path to config/promoter.yaml")
    args = ap.parse_args()

    # load and apply configuration (safe fallback to defaults)
    try:
        cfg = load_config(args.config)
    except Exception:
        cfg = None
    apply_config(cfg)

    import asyncio
    asyncio.run(run_daily(args.date, args.github_token))


if __name__ == "__main__":
    main()