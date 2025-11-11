# /web/api.py
# FastAPI backend for LLM Trend Radar (replacing Streamlit UI)
# - Endpoints: /health, /search, /chat
# - Reuses scripts.search_runner if available; falls back to FTS-only
# - Works with your existing SQLite (data/trend_radar.db) and Ollama

import os, re, sqlite3, time, sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import HTMLResponse

# Ensure project root is on sys.path so `scripts.*` imports always work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DB = os.environ.get("TREND_RADAR_DB", str(PROJECT_ROOT / "data" / "trend_radar.db"))
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/api/chat")

# ========= try import your advanced search (defensive) =========
SEARCH_OK = False
SEARCH_ERR = ""
sr_search_fts = sr_search_faiss = sr_search_hybrid = None
sr_compose_prompt = sr_ask_ollama = None
try:
    import scripts.search_runner as SR  # single import so partial availability won't crash
    sr_search_fts = getattr(SR, "search_fts", None)
    sr_search_faiss = getattr(SR, "search_faiss", None)
    sr_search_hybrid = getattr(SR, "search_hybrid", None)
    sr_compose_prompt = getattr(SR, "compose_prompt", None)
    sr_ask_ollama = getattr(SR, "ask_ollama", None)
    # flag as available if at least one retriever exists
    SEARCH_OK = any([sr_search_fts, sr_search_faiss, sr_search_hybrid])
except Exception as e:
    SEARCH_OK = False
    SEARCH_ERR = repr(e)

# ========= helpers =========
def _connect(db: str):
    if not os.path.exists(db):
        raise FileNotFoundError(f"DB not found: {db}")
    return sqlite3.connect(db)

def _sanitize_fts_query(q: str) -> str:
    # Avoid fts5 syntax errors
    q = re.sub(r'["\'*?(){}:\[\]\\]', " ", q)
    q = re.sub(r'\b(AND|OR|NOT)\b', " ", q, flags=re.IGNORECASE)
    q = re.sub(r'\s+', " ", q).strip()
    terms = [t for t in q.split(" ") if t]
    return " AND ".join(terms) if terms else ""

def _fts_only_search(db: str, query: str, k: int) -> List[Dict[str, Any]]:
    q = _sanitize_fts_query(query)
    if not q:
        return []
    con = _connect(db)
    sql = """
    SELECT c.doc_id,
           bm25(fts_chunk) AS rank,
           c.rowid,
           snippet(fts_chunk, -1, '', '', ' … ', 80) AS snippet,
           COALESCE(d.title, ''), COALESCE(d.source_type, '?')
    FROM fts_chunk
    JOIN doc_chunk c ON c.rowid = fts_chunk.rowid
    LEFT JOIN doc d ON d.id = c.doc_id
    WHERE fts_chunk MATCH ?
    ORDER BY rank
    LIMIT ?;
    """
    rows = con.execute(sql, (q, k)).fetchall()
    con.close()
    hits = []
    for doc_id, rank, rowid, snippet, title, src in rows:
        score = 1.0 / (1.0 + max(rank, 0.0))
        hits.append({"id": int(doc_id), "score": float(score),
                     "snippet": snippet or "", "title": title or "", "source": src or "?"})
    return hits

def _fetch_doc_title_source(db: str, doc_id: int) -> Tuple[str, str]:
    con = _connect(db)
    row = con.execute("SELECT COALESCE(title,''), COALESCE(source_type,'?') FROM doc WHERE id=?",
                      (doc_id,)).fetchone()
    con.close()
    return (row[0] or f"Doc {doc_id}", row[1] or "?") if row else (f"Doc {doc_id}", "?")

def _compose_prompt(query: str, hits: List[Dict[str, Any]], max_chars: int = 2200):
    ctx, picked, remain = [], [], max_chars
    for h in hits:
        chunk = (h.get("snippet") or "").strip()
        if not chunk:
            continue
        block = f"Source doc_id={h['id']}\n{chunk}\n"
        if len(block) <= remain:
            ctx.append(block); picked.append(h); remain -= len(block)
        if remain <= 200:
            break
    system = ("You are a helpful assistant. Use the following context to answer concisely. "
              "Cite sources as [S1], [S2] using the order of the provided context.")
    context_text = "\n---\n".join(ctx) if ctx else "No context."
    user_prompt = f"{system}\n\n# Question\n{query}\n\n# Context\n{context_text}\n\n# Answer with sources."
    return user_prompt, picked

def _ask_ollama(model: str, prompt: str) -> str:
    import requests
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False}
    r = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "message" in data:
        return data["message"].get("content", "").strip()
    return data.get("response", "").strip()


# Fallback hybrid combiner if advanced hybrid not available
def _hybrid_fallback(db: str, query: str, topk: int, alpha: float) -> List[Dict[str, Any]]:
    """
    Blend FAISS and FTS results when scripts.search_runner.search_hybrid is missing.
    Will gracefully degrade to whichever method is available.
    """
    # get candidate lists
    faiss_hits = []
    fts_hits = []
    try:
        if sr_search_faiss:
            faiss_hits = sr_search_faiss(db, query, topk=topk)
    except Exception:
        faiss_hits = []
    try:
        fts_hits = sr_search_fts(db, query, topk=topk) if sr_search_fts else _fts_only_search(db, query, topk)
    except Exception:
        fts_hits = _fts_only_search(db, query, topk)

    # if one side is empty, return the other
    if not faiss_hits and not fts_hits:
        return []
    if not faiss_hits:
        return fts_hits[:topk]
    if not fts_hits:
        return faiss_hits[:topk]

    # index by doc id and min-max normalize each list
    def _norm(scores):
        if not scores:
            return {}
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        if hi <= lo:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    f_dict = {h["id"]: float(h.get("score", 0.0)) for h in fts_hits}
    v_dict = {h["id"]: float(h.get("score", 0.0)) for h in faiss_hits}
    f_norm = _norm(f_dict)
    v_norm = _norm(v_dict)

    ids = set(f_norm) | set(v_norm)
    combined: List[Tuple[int, float]] = []
    for i in ids:
        sf = f_norm.get(i, 0.0)
        sv = v_norm.get(i, 0.0)
        s = (1.0 - alpha) * sf + alpha * sv
        combined.append((i, s))
    combined.sort(key=lambda x: x[1], reverse=True)
    top_ids = [i for i, _ in combined[:topk]]

    # build merged hits with snippet/title/source if available
    merged = {h["id"]: dict(h) for h in (fts_hits + faiss_hits)}
    result = [merged[i] for i in top_ids if i in merged]
    # ensure score reflects blended score
    for i, s in combined[:topk]:
        if i in merged:
            merged[i]["score"] = float(s)
    return result

def retrieve(db: str, query: str, mode: str, topk: int, alpha: float):
    if SEARCH_OK:
        try:
            if mode == "fts":
                if sr_search_fts:
                    hits = sr_search_fts(db, query, topk=topk)
                else:
                    hits = _fts_only_search(db, query, topk)
            elif mode == "faiss":
                if sr_search_faiss:
                    hits = sr_search_faiss(db, query, topk=topk)
                else:
                    # graceful downgrade
                    hits = _fts_only_search(db, query, topk)
            else:  # hybrid
                if sr_search_hybrid:
                    hits = sr_search_hybrid(db, query, topk=topk, alpha=alpha)
                else:
                    hits = _hybrid_fallback(db, query, topk=topk, alpha=alpha)
        except Exception:
            # last-resort fallback
            hits = _fts_only_search(db, query, topk)
    else:
        hits = _fts_only_search(db, query, topk)

    # backfill title/source if missing
    for h in hits:
        if not h.get("title") or not h.get("source"):
            t, s = _fetch_doc_title_source(db, h["id"])
            h["title"] = h.get("title") or t
            h["source"] = h.get("source") or s
    return hits

def build_prompt(query: str, hits: List[Dict[str, Any]]):
    if SEARCH_OK:
        try:
            return sr_compose_prompt(query, hits)
        except Exception:
            pass
    return _compose_prompt(query, hits)

def ask_llm(model: str, prompt: str):
    if SEARCH_OK:
        try:
            return sr_ask_ollama(model=model, prompt=prompt)
        except Exception:
            pass
    return _ask_ollama(model=model, prompt=prompt)

def check_db(db: str) -> Dict[str, Any]:
    ok = os.path.exists(db)
    info = {"db_path": db, "exists": ok, "tables_ok": False, "docs": 0, "chunks": 0, "fts_chunks": 0}
    if not ok:
        return info
    con = sqlite3.connect(db)
    tables = set(r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'"))
    need = {"doc", "doc_chunk", "fts_chunk"}
    info["tables_ok"] = need.issubset(tables)
    if info["tables_ok"]:
        info["docs"] = con.execute("SELECT COUNT(*) FROM doc").fetchone()[0]
        info["chunks"] = con.execute("SELECT COUNT(*) FROM doc_chunk").fetchone()[0]
        info["fts_chunks"] = con.execute("SELECT COUNT(*) FROM fts_chunk").fetchone()[0]
    con.close()
    return info

# ========= FastAPI app =========
app = FastAPI(title="LLM Trend Radar API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class SearchReq(BaseModel):
    query: str
    mode: str = "hybrid"   # hybrid | fts | faiss
    topk: int = 5
    alpha: float = 0.5
    db: Optional[str] = None

class ChatReq(SearchReq):
    model: str = DEFAULT_MODEL

@app.get("/health")
def health(db: Optional[str] = None):
    info = check_db(db or DEFAULT_DB)
    info["search_ok"] = SEARCH_OK
    if not SEARCH_OK:
        info["search_error"] = SEARCH_ERR
    return info

@app.post("/search")
def api_search(req: SearchReq):
    db = req.db or DEFAULT_DB
    t0 = time.time()
    try:
        hits = retrieve(db, req.query, req.mode, req.topk, req.alpha)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed: {e}")
    t1 = time.time()
    return {"hits": hits, "search_ms": int((t1 - t0) * 1000)}

@app.post("/chat")
def api_chat(req: ChatReq):
    db = req.db or DEFAULT_DB
    t0 = time.time()
    try:
        hits = retrieve(db, req.query, req.mode, req.topk, req.alpha)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed: {e}")
    t1 = time.time()
    prompt, picked = build_prompt(req.query, hits)
    try:
        answer = ask_llm(req.model, prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"llm failed: {e}")
    t2 = time.time()
    return {
        "answer": answer,
        "sources": picked,
        "search_ms": int((t1 - t0) * 1000),
        "gen_ms": int((t2 - t1) * 1000),
        "total_ms": int((t2 - t0) * 1000),
        "search_ok": SEARCH_OK,
    }

# very small html demo (optional)
@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LLM Trend Radar</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
</style>
</head>
<body class="bg-slate-50 text-slate-900">
  <div class="max-w-5xl mx-auto p-6">
    <div class="flex items-center justify-between mb-6">
      <h1 class="text-2xl font-bold">LLM Trend Radar</h1>
      <div id="health" class="text-sm text-slate-500"></div>
    </div>

    <div class="bg-white shadow-sm rounded-lg p-4 mb-4">
      <div class="flex gap-2">
        <input id="q" class="flex-1 border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="Ask your question...">
        <select id="mode" class="border border-slate-300 rounded-lg px-2 py-2">
          <option value="hybrid">hybrid</option>
          <option value="fts">fts</option>
          <option value="faiss">faiss</option>
        </select>
        <button id="askBtn" class="bg-indigo-600 text-white rounded-lg px-4 py-2 hover:bg-indigo-700">Ask</button>
      </div>
      <div class="grid grid-cols-1 md:grid-cols-4 gap-3 mt-3">
        <div>
          <label class="text-xs text-slate-500">Top-K</label>
          <input id="topk" type="number" min="1" max="20" value="5" class="w-full border border-slate-300 rounded-lg px-2 py-1">
        </div>
        <div>
          <label class="text-xs text-slate-500">Hybrid α</label>
          <input id="alpha" type="number" step="0.05" min="0" max="1" value="0.5" class="w-full border border-slate-300 rounded-lg px-2 py-1">
        </div>
        <div>
          <label class="text-xs text-slate-500">Model</label>
          <input id="model" value="qwen2.5:14b" class="w-full border border-slate-300 rounded-lg px-2 py-1">
        </div>
        <div>
          <label class="text-xs text-slate-500">DB (optional)</label>
          <input id="db" placeholder="leave empty to use default" class="w-full border border-slate-300 rounded-lg px-2 py-1">
        </div>
      </div>
    </div>

    <div id="answer" class="bg-white shadow-sm rounded-lg p-4 mono whitespace-pre-wrap"></div>

    <h3 class="mt-6 mb-2 font-semibold">Sources</h3>
    <div id="sources" class="grid grid-cols-1 md:grid-cols-2 gap-3"></div>
  </div>

<script>
async function loadHealth() {
  const r = await fetch('/health');
  const j = await r.json();
  const h = document.getElementById('health');
  const ok = j.exists && j.tables_ok;
  h.innerHTML = `
    <span class="${ok ? 'text-emerald-600' : 'text-rose-600'}">${ok ? 'DB ready' : 'DB missing'}</span>
    · docs ${j.docs||0} · chunks ${j.chunks||0} · fts ${j.fts_chunks||0}
    · search_runner ${j.search_ok ? 'ON' : 'OFF'}
    <span class="ml-1 text-slate-400">(${j.db_path})</span>
  `;
}

function sourceCard(i, h) {
  const t = h.title || ('Doc '+h.id);
  const s = h.source || '?';
  const sc = (h.score||0).toFixed(4);
  const snip = (h.snippet||'').replaceAll('<b>','').replaceAll('</b>','');
  return `
    <div class="bg-white rounded-lg shadow-sm p-3">
      <div class="text-sm font-medium">${i}. ${t}</div>
      <div class="text-xs text-slate-500 mb-1">${s} · score=${sc}</div>
      <div class="text-sm text-slate-700 line-clamp-5">${snip}</div>
    </div>
  `;
}

async function ask() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const mode = document.getElementById('mode').value;
  const topk = parseInt(document.getElementById('topk').value || '5', 10);
  const alpha = parseFloat(document.getElementById('alpha').value || '0.5');
  const model = document.getElementById('model').value || 'qwen2.5:14b';
  const db = document.getElementById('db').value;

  const resp = await fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ query:q, mode, topk, alpha, model, db: db || null })
  });
  const js = await resp.json();
  const ans = document.getElementById('answer');
  if (resp.ok) {
    ans.textContent = `### Answer\\n${js.answer}\\n\\n—\\n(search ${js.search_ms}ms, gen ${js.gen_ms}ms, total ${js.total_ms}ms)`;
    const src = document.getElementById('sources');
    src.innerHTML = (js.sources||[]).map((h,idx)=>sourceCard(idx+1,h)).join('');
  } else {
    ans.textContent = 'ERROR: ' + (js.detail || JSON.stringify(js));
  }
}

document.getElementById('askBtn').addEventListener('click', ask);
document.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && e.metaKey===false && e.ctrlKey===false){ ask(); }});
loadHealth();
</script>
</body>
</html>"""