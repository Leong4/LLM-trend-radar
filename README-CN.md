
# LLM 技术趋势雷达系统

> 从 **arXiv + GitHub** 自动采集 → 清洗切块 → 建索引（FTS/FAISS）→ 趋势雷达打分与状态机（IGNORE / PROBATION / FULL + UCB 探索）→ 基于 RAG 的问答服务（FastAPI + 本地 Ollama）。

[英文文档 / English README](./README-EN.md)
---

## ✨ 功能亮点

- **数据采集**：arXiv（英文优先，PDF-first 解析，失败回退摘要）、GitHub（聚焦 LLM/RAG 主题）。
- **清洗预处理**：语言检测、元数据入库、PDF 摘要/正文解析、切块、停用词处理。
- **检索索引**：SQLite FTS5（BM25）＋ 可选 FAISS（向量检索），支持 **FTS / FAISS / Hybrid**。
- **趋势雷达**：三态状态机 **IGNORE / PROBATION / FULL**，按日写入 `signals`，内置 **UCB 探索**、试用期与 TTL 回收。
- **问答服务**：FastAPI 提供 `/health`、`/search`、`/chat`，默认调用本地 **Ollama qwen2.5:14b**。
- **报表与监控**：周报 Markdown、迁移审计 CSV、（可选）指标 watcher 脚本。

---

## 🗂️ 目录结构（关键部分）

```text
project1/
├─ config/
│  └─ promoter.yaml           # 趋势雷达参数（阈值/配额/白名单/窗口等）
├─ data/
│  └─ trend_radar.db          # SQLite 数据库（自动创建/更新）
├─ reporting/
│  └─ weekly_YYYYWW.md        # 周报输出（示例）
├─ scripts/
│  ├─ ingest_runner.py        # 采集入口（arXiv/GitHub）
│  ├─ preprocess_runner.py    # 预处理入口（PDF-first、切块、建索引）
│  ├─ search_runner.py        # FTS/FAISS/Hybrid 检索封装 + 组合 Prompt
│  └─ qa_runner.py            # CLI 问答自检（与 Web 端一致的链路）
├─ pipeline/
│  ├─ daily_promote.py        # 日批：计算 signals + 状态机迁移（写 transitions）
│  └─ weekly_allocate.py      # 周配额：按簇分配 FULL/探索名额
├─ sql/
│  └─ 001_promoter.sql        # promoter 相关表的建表脚本（items/signals/transitions/labels）
├─ web/
│  └─ api.py                  # FastAPI 后端（/health /search /chat + 简易前端）
└─ README.md
```

> 注：最初的 Streamlit UI 已废弃；保留 `web/api.py`（FastAPI）作为展示与集成入口。

---

## ⚙️ 环境准备

- macOS / Linux（当前工程环境为 macOS）
- Python **3.10+**（推荐）
- 本地 LLM：**Ollama**（默认模型 `qwen2.5:14b`）

安装依赖（示例）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install fastapi "uvicorn[standard]" pydantic
pip install numpy sqlite-utils
# 如果需要向量检索：
pip install faiss-cpu   # 安装失败可先用 FTS（系统会自动降级）
```

> 向量索引为可选；缺失/安装失败时系统自动回退到 FTS 检索。

---

## 🚀 快速开始（端到端）

> 建议在**项目根目录**执行；DB 路径将按绝对路径解析，避免“找不到库”。

### 0) 准备配置

`config/promoter.yaml`（示例，已内置）：

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

### 1) 采集（arXiv + GitHub）

```bash
# 拉取近 N 天 arXiv + GitHub（可在 ingest_runner 里调节 topic/日期）
python -m scripts.ingest_runner --sources arxiv,github --days 14 --min_citations 0 --min_stars 0
```

> 早期曾尝试 HuggingFace/npm，此处已移除；聚焦 **arXiv + GitHub** 的高信噪组合。

### 2) 预处理（PDF-first、切块、索引）

```bash
# 常规 PDF-first（失败回退摘要），并导出统计
python -m scripts.preprocess_runner --pdf-first --export-stats

# 如已有大量 arXiv 仅落了摘要（lite），可回填 PDF：
python -m scripts.preprocess_runner --pdf-backfill --export-stats
```

完成后，DB 中应包含：

- `doc` / `doc_chunk`（文档与切片）
- `fts_chunk`（FTS5 虚表）
- （可选）FAISS 索引文件（由 `search_runner` 管理）

### 3) 趋势雷达（日批 / 周配额）

**数据库迁移**（首次或结构变更时）：

```bash
sqlite3 data/trend_radar.db < sql/001_promoter.sql
```

**日批：计算 signals + 状态机迁移（写 transitions）**

```bash
python -m pipeline.daily_promote --config config/promoter.yaml
```

**周配额：按簇分配 FULL / 探索名额 + 试用期**

```bash
python -m pipeline.weekly_allocate --config config/promoter.yaml
```

**周报**

```bash
python -m reporting.weekly --out reporting/weekly_$(date +%G%V).md
```

三态机要点：

- **IGNORE / PROBATION / FULL**
- **分段权重**：`Prior / Velocity / Semantic / Reputation` 随 `age_days` 动态调整
- **UCB 探索**：给高不确定性的新对象探索名额
- **试用期与回收**：`FULL` 连续 7 天低分降级、`PROBATION` 30 天无改善转 `IGNORE`

### 4) 启动问答服务（FastAPI + Ollama）

```bash
# 强烈建议用绝对路径指定 DB
export TREND_RADAR_DB="$(pwd)/data/trend_radar.db"
export OLLAMA_MODEL="qwen2.5:14b"
# 确保从项目根启动（让 scripts.* 位于 PYTHONPATH）
python -m uvicorn web.api:app --reload --port 8000
```

打开：

- 健康检查：<http://localhost:8000/health>（显示 docs/chunks/fts 数量与 search_runner 状态）
- 简易网页：<http://localhost:8000/>（参数面板 + Sources 卡片）
- Swagger：<http://localhost:8000/docs>

> `search_ok:false` 多半是 `PYTHONPATH` 或 `scripts.search_runner` 缺函数；我们已做防御性降级：
> - **Hybrid / FAISS 失效 → 自动回退 FTS**
> - 少函数不会“全盘失败”，UI 仍可用

### 5) CLI 自检

```bash
# FTS 片段快速验证（应能看到 snippet 文本）
sqlite3 "$TREND_RADAR_DB" "
SELECT d.title, snippet(fts_chunk, -1, '', '', ' … ', 64)
FROM fts_chunk
JOIN doc_chunk c ON c.rowid=fts_chunk.rowid
JOIN doc d ON d.id=c.doc_id
LIMIT 3;"
```

```bash
# CLI 问答（与 Web 端一致的链路）
python -m scripts.qa_runner --mode fts --topk 5 -q "What are current hot RAG retrieval ideas?"
```

---

## 🔧 配置与接口契约

- **环境变量**
  - `TREND_RADAR_DB`：SQLite 绝对路径（推荐）
  - `OLLAMA_MODEL`：默认 `qwen2.5:14b`
  - `OLLAMA_ENDPOINT`：默认 `http://localhost:11434/api/chat`
- **配置文件**
  - `config/promoter.yaml`：阈值、UCB 系数、配额、生命周期、白名单、窗口等
- **接口契约（示例位置）**
  - 采集 → 入库：`schemas/paper.py`（Pydantic）＋ `db.models`（SQLAlchemy/SQLite）
  - 检索 → 问答：`scripts/search_runner.py` 统一返回字段：
    ```json
    {"id": doc_id, "score": float, "title": str, "source": str, "snippet": str}
    ```

---

## 🗃️ 数据库表（核心）

- **内容层**
  - `doc(id, source_type, title, ...)`
  - `doc_chunk(id, doc_id, text, ...)`
  - `fts_chunk`（FTS5 虚表，对 `doc_chunk` 建的全文索引）
- **雷达层**
  - `items(id, source, created_at, state, last_score, last_ucb, obs_days, cluster_id, meta_json)`
  - `signals(id, day, prior, velocity, semantic, reputation, score, ucb)` ← 按日写
  - `transitions(id, ts, from_state, to_state, reason, details)`
  - `labels(id, ts, label, note)` ← 人工/下游反馈闭环

---

## 📊 报表与监控（可选）

- 周报：`reporting/weekly_YYYYWW.md`
- 迁移审计：`exports/audit_transitions.csv`
- 指标面板：`exports/metrics_daily.csv`
- 监控脚本（示例）：
  - **UCB P90 连续两天下破 0.6**：`watch_ucb_probation.py`
  - **近 7 天 `trial_review_demote` 降级 > 升级**：见 `docs/automation.md` 的 SQL/脚本示例

---

## 🧩 常见问题（FAQ）

- **`/health` 显示 `search_ok:false`**
  - 从项目根启动：`python -m uvicorn web.api:app --reload`
  - 或设置 `PYTHONPATH=.`  
  - 缺少 `search_hybrid` 不影响使用（内置 Hybrid fallback）
- **UI 显示“无可用上下文 / 耗时 0”**
  - 大概率连错 DB 或库里没 `fts_chunk`；请检查 `docs / chunks / fts_chunks` 数量
  - 确保已经运行 `preprocess_runner --pdf-first` 并成功建索引
  - FTS 查询语法会自动清洗非法字符（如 `()"?:` 等）
- **FAISS 安装失败**
  - 先用 FTS-only，等依赖 OK 再开向量检索；Hybrid 会自动回退
- **arXiv 全是 lite（仅摘要）**
  - 使用 `--pdf-backfill` 重跑预处理，覆盖为 PDF-first 解析

---

## 📜 许可与合规

- 遵循 arXiv / GitHub 站点条款与 `robots.txt`；控制抓取频率；缓存 PDF，不对外分发；仅用于研究与检索
- 如需对外提供服务，请自行增加频控、缓存与合规说明

---

## 🛣️ Roadmap（下一步）

- 加强 GitHub 先验与趋势信号（tests/CI、活跃度、issues/PR 唯一用户数）
- 主题簇自动更新与每周配额自适应
- 前端独立化（React/Vue），对接 `/search` `/chat`
- 指标面板（Top‑K 命中率、探索命中率、AUC）

---

## 👤 作者

- Henry Leong（仓库所有者）

---

## 🔁 一键复现（最短路径）

```bash
# 1) 采集
python -m scripts.ingest_runner --sources arxiv,github --days 14 --min_citations 0 --min_stars 0

# 2) 预处理（PDF-first）
python -m scripts.preprocess_runner --pdf-first --export-stats

# 3) 迁移（首次）
sqlite3 data/trend_radar.db < sql/001_promoter.sql

# 4) 日批 + 周配额 + 周报
python -m pipeline.daily_promote --config config/promoter.yaml
python -m pipeline.weekly_allocate --config config/promoter.yaml
python -m reporting.weekly --out reporting/weekly_$(date +%G%V).md

# 5) 启动服务
export TREND_RADAR_DB="$(pwd)/data/trend_radar.db"
export OLLAMA_MODEL="qwen2.5:14b"
python -m uvicorn web.api:app --reload --port 8000

# 6) 打开 http://localhost:8000/health 与 /docs
```