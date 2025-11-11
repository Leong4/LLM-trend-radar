# LLM Tech Trend Radar

> End‑to‑end pipeline: **arXiv + GitHub** ingestion → cleaning & chunking → indexing (SQLite **FTS5** / optional **FAISS**) → **Trend Radar** with a 3‑state machine (**IGNORE / PROBATION / FULL**) + **UCB** exploration → **RAG** Q&A (FastAPI + local **Ollama**).

[中文文档 / Chinese README](./README-CN.md)

---

## ✨ Highlights

- **Data ingestion**: arXiv (English‑first, **PDF‑first** with abstract fallback), GitHub (LLM/RAG topics).
- **Preprocess**: language detection, metadata, PDF parsing, chunking, stopword handling.
- **Search**: SQLite **FTS5** (BM25) + optional **FAISS**; supports **FTS / FAISS / Hybrid**.
- **Trend radar**: 3‑state machine, daily **signals**, **UCB** exploration, trial & TTL reclaim, weekly quota.
- **Q&A**: FastAPI endpoints **/health**, **/search**, **/chat** using local **Ollama qwen2.5:14b** by default.
- **Reporting & monitoring**: weekly Markdown report, transition audit CSV, simple watcher scripts.

> The former Streamlit UI is deprecated. Use **FastAPI** (`web/api.py`) for demo & integration.

---

## 🗂️ Repository Layout (key parts)

```text
project1/
├─ config/
│  └─ promoter.yaml           # Radar thresholds/quotas/whitelists/windows
├─ data/
│  └─ trend_radar.db          # SQLite database (auto created/updated)
├─ reporting/
│  └─ weekly_YYYYWW.md        # Weekly report output (example)
├─ scripts/
│  ├─ ingest_runner.py        # Ingestion entry (arXiv/GitHub)
│  ├─ preprocess_runner.py    # Preprocess entry (PDF-first, chunk, index)
│  ├─ search_runner.py        # FTS/FAISS/Hybrid + prompt composer
│  └─ qa_runner.py            # CLI QA self-check (same chain as Web)
├─ pipeline/
│  ├─ daily_promote.py        # Daily signals + state transitions
│  └─ weekly_allocate.py      # Weekly quota per cluster (FULL & explore)
├─ sql/
│  └─ 001_promoter.sql        # Items/signals/transitions/labels tables
├─ web/
│  └─ api.py                  # FastAPI backend (/health /search /chat + minimal UI)
└─ README.md
```

---

## ⚙️ Requirements

- macOS / Linux
- Python **3.10+** recommended
- Local LLM via **Ollama** (default model: `qwen2.5:14b`)

Install deps (example):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install fastapi "uvicorn[standard]" pydantic
pip install numpy sqlite-utils
# Optional vector search:
pip install faiss-cpu  # if this fails, the system falls back to FTS automatically
```

> Vector search is optional; the system gracefully degrades to FTS when FAISS is unavailable.

---

## 🚀 Quick Start (end‑to‑end)

Run all commands from the **project root**. The DB path is resolved absolutely to avoid “not found”.

### 0) Config

`config/promoter.yaml` (example shipped):

```yaml
thresholds:
  tau_prob: 0.30
  tau_full_young: 0.65
  tau_full_mid: 0.60
  tau_full_old: 0.55
  tau_demote: 0.45
ucb:
  c: 1.0
quotas:
  full_per_cluster: 20
  explore_per_cluster: 5
lifecycle:
  full_trial_days: 14
  probation_ttl_days: 30
whitelists:
  labs: ["DeepMind", "FAIR", "MSR", "Tsinghua AIR", "CMU LTI"]
features:
  github:
    readme_len_max: 8000
    commit_window_days: 14
    delta_star_window_days: 7
  arxiv:
    mention_window_days: 7
```

### 1) Ingestion (arXiv + GitHub)

```bash
python -m scripts.ingest_runner --sources arxiv,github --days 14 --min_citations 0 --min_stars 0
```

> HF/npm sources were removed to keep a high‑SNR core (arXiv + GitHub).

### 2) Preprocess (PDF‑first, chunk, index)

```bash
# PDF-first (fallback to abstract on failure) + export stats
python -m scripts.preprocess_runner --pdf-first --export-stats

# Backfill PDF for existing arXiv "lite" docs (abstract-only):
python -m scripts.preprocess_runner --pdf-backfill --export-stats
```

After this, the DB contains:

- `doc` / `doc_chunk` (documents & chunks)
- `fts_chunk` (FTS5 virtual table)
- (Optional) FAISS index files (managed by `search_runner`)

### 3) Trend Radar (daily / weekly)

**DB migration** (first time or when schema changes):

```bash
sqlite3 data/trend_radar.db < sql/001_promoter.sql
```

**Daily: signals + state transitions**

```bash
python -m pipeline.daily_promote --config config/promoter.yaml
```

**Weekly: allocate FULL / explore per cluster**

```bash
python -m pipeline.weekly_allocate --config config/promoter.yaml
```

**Weekly report**

```bash
python -m reporting.weekly --out reporting/weekly_$(date +%G%V).md
```

State machine essentials:

- **IGNORE / PROBATION / FULL**
- **Age‑aware weights** for `Prior / Velocity / Semantic / Reputation`
- **UCB exploration** for new & uncertain items
- **Trial & TTL** reclaim (demote FULL on prolonged low scores; TTL PROBATION → IGNORE)

### 4) Start Q&A service (FastAPI + Ollama)

```bash
# Use an absolute DB path for stability
export TREND_RADAR_DB="$(pwd)/data/trend_radar.db"
export OLLAMA_MODEL="qwen2.5:14b"
# Start from project root so scripts.* are on PYTHONPATH
python -m uvicorn web.api:app --reload --port 8000
```

Open:

- Health: <http://localhost:8000/health> (docs/chunks/fts counts, search_runner ON/OFF)
- Minimal UI: <http://localhost:8000/>
- Swagger: <http://localhost:8000/docs>

> If `/health` reports `search_ok:false`, it’s usually `PYTHONPATH` or missing functions in `scripts.search_runner`. We added defensive fallbacks:
> - **Hybrid/FAISS → FTS** auto‑fallback  
> - Missing functions no longer break the API; the UI remains usable.

### 5) CLI sanity checks

```bash
# Quick FTS snippet check (should show snippet text)
sqlite3 "$TREND_RADAR_DB" "
SELECT d.title, snippet(fts_chunk, -1, '', '', ' … ', 64)
FROM fts_chunk
JOIN doc_chunk c ON c.rowid=fts_chunk.rowid
JOIN doc d ON d.id=c.doc_id
LIMIT 3;"
```

```bash
# CLI QA (same chain as Web)
python -m scripts.qa_runner --mode fts --topk 5 -q "What are current hot RAG retrieval ideas?"
```

---

## 🔧 Config & Contracts

- **Env vars**
  - `TREND_RADAR_DB`: absolute path to SQLite (recommended)
  - `OLLAMA_MODEL`: default `qwen2.5:14b`
  - `OLLAMA_ENDPOINT`: default `http://localhost:11434/api/chat`
- **Config file**
  - `config/promoter.yaml`: thresholds, UCB, quotas, lifecycles, whitelists, windows.
- **Interfaces**
  - Ingestion → DB: Pydantic schemas (e.g., `schemas/paper.py`) + SQLAlchemy/SQLite models (`db.models`).
  - Search → QA: `scripts/search_runner.py` returns unified hits:
    ```json
    {"id": doc_id, "score": 0.0, "title": "str", "source": "str", "snippet": "str"}
    ```

---

## 🗃️ Core Tables

- **Content**
  - `doc(id, source_type, title, ...)`
  - `doc_chunk(id, doc_id, text, ...)`
  - `fts_chunk` (FTS5 virtual table on `doc_chunk`)
- **Radar**
  - `items(id, source, created_at, state, last_score, last_ucb, obs_days, cluster_id, meta_json)`
  - `signals(id, day, prior, velocity, semantic, reputation, score, ucb)`
  - `transitions(id, ts, from_state, to_state, reason, details)`
  - `labels(id, ts, label, note)`

---

## 📊 Reports & Monitoring (optional)

- Weekly report: `reporting/weekly_YYYYWW.md`
- Transition audit: `exports/audit_transitions.csv`
- Metrics: `exports/metrics_daily.csv`
- Watchers (examples):
  - `watch_ucb_probation.py`: **p90(UCB, PROBATION) < 0.6** for **two consecutive days**
  - SQL/CLI snippets for “7‑day `trial_review_demote` demotions > promotions”

---

## 🧩 FAQ

- **`/health` says `search_ok:false`**
  - Start from project root: `python -m uvicorn web.api:app --reload` (or set `PYTHONPATH=.`).
  - Missing `search_hybrid` is fine: Hybrid fallback is built‑in.
- **UI shows “no contexts / 0ms search”**
  - Likely wrong DB path or missing `fts_chunk`. Check `docs/chunks/fts_chunks` counts.
  - Ensure you ran `preprocess_runner --pdf-first` to build the FTS index.
  - The FTS query text is auto‑sanitized (removes `()"?:` etc.) to avoid query errors.
- **FAISS install fails**
  - Use FTS‑only for now; Hybrid gracefully falls back.
- **arXiv entries are “lite” (abstract‑only)**
  - Run `--pdf-backfill` to overwrite lite with PDF‑first parsing.

---

## 📜 License & Compliance

- Respect arXiv/GitHub ToS and `robots.txt`. Rate‑limit requests. Cache PDFs locally; do **not** redistribute.
- When exposing a public service, add frequency limiting, caching, and compliance notices.

---

## 🛣️ Roadmap

- Stronger GitHub prior/velocity signals (tests/CI, contributor activity, distinct issue/PR users).
- Automatic cluster maintenance & quota adaptation.
- Standalone React/Vue frontend over `/search` and `/chat`.
- Metrics dashboard (Top‑K hit rate, exploration hit rate, AUC).

---

## 👤 Author

- Henry Leong (repo owner)

---

## 🔁 One‑Shot Repro

```bash
# 1) Ingest
python -m scripts.ingest_runner --sources arxiv,github --days 14 --min_citations 0 --min_stars 0

# 2) Preprocess (PDF-first)
python -m scripts.preprocess_runner --pdf-first --export-stats

# 3) Migration (first time)
sqlite3 data/trend_radar.db < sql/001_promoter.sql

# 4) Daily + Weekly + Report
python -m pipeline.daily_promote --config config/promoter.yaml
python -m pipeline.weekly_allocate --config config/promoter.yaml
python -m reporting.weekly --out reporting/weekly_$(date +%G%V).md

# 5) Start API
export TREND_RADAR_DB="$(pwd)/data/trend_radar.db"
export OLLAMA_MODEL="qwen2.5:14b"
python -m uvicorn web.api:app --reload --port 8000

# 6) Open http://localhost:8000/health  and  /docs
```
