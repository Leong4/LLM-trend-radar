# llm_radar/utils/conf.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None  # 允许没有 YAML 时走默认

DEFAULTS: Dict[str, Any] = {
    "thresholds": {
        "tau_prob": 0.30, "tau_full_young": 0.65, "tau_full_mid": 0.60,
        "tau_full_old": 0.55, "tau_demote": 0.45,
    },
    "lifecycle": {"full_trial_days": 14, "probation_ttl_days": 30},
    "ucb": {"c": 1.0},
    "quotas": {"full_per_cluster": 20, "explore_per_cluster": 5},
    "whitelists": {"labs": []},
    "keywords": {"llm_rag": []},
    "features": {
        "github": {
            "readme_len_max": 8000, "delta_star_window_days": 7,
            "commit_window_days": 14, "uniq_issue_authors_window_days": 14,
            "star_cap": 200, "actor_cap": 30,
        },
        "arxiv": {"mention_window_days": 7},
    },
    "preprocess": {
        "ucb_threshold": 0.70, "chunk_size": 1200, "overlap": 200,
        "pdf": {"strip_references": True},
    },
}

def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: str | None) -> Dict[str, Any]:
    if not path or not yaml:
        return DEFAULTS
    p = Path(path)
    if not p.exists():
        return DEFAULTS
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return deep_update(DEFAULTS, data)