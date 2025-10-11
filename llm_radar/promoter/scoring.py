from __future__ import annotations
import math

def normalize(x: float | None, lo: float, hi: float, fallback: float = 0.1) -> float:
    if x is None:
        return fallback
    if hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    return max(0.0, min(1.0, v))

def blend_by_age(prior: float, velocity: float, semantic: float, reputation: float, age_days: int) -> float:
    if age_days <= 14:
        w = (0.1, 0.4, 0.4, 0.1)
    elif age_days <= 90:
        w = (0.25, 0.25, 0.35, 0.15)
    else:
        w = (0.3, 0.1, 0.2, 0.4)
    s = w[0]*prior + w[1]*velocity + w[2]*semantic + w[3]*reputation
    return max(0.0, min(1.0, s))

def ucb(m: float, n: int, T: int, c: float = 1.0) -> float:
    """
    m: 已标准化(0..1)的 Velocity 均值
    n: 观测天数
    T: 系统运行天数
    c: 探索系数
    """
    bonus = c * math.sqrt(max(0.0, math.log(max(2, T)) / max(1, n)))
    raw = m + bonus
    # 理论最大值发生在 m=1、n=1，此时 raw_max = 1 + bonus_max
    bonus_max = c * math.sqrt(math.log(max(2, T)))
    denom = 1.0 + bonus_max
    u = raw / denom
    return max(0.0, min(1.0, u))