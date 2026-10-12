"""Daily top-K queue evaluation, random baseline and paired bootstrap.

Definitions for an approval day d (day = calendar date of prediction_time as
stored in the data, i.e. the local timestamps of the Olist export; no timezone
conversion is applied):

    n_d  number of evaluable orders approved that day (the cohort passed in)
    m_d  number of late orders among them
    k_d  = min(K, n_d)                       orders reviewed that day
    h_d  late orders among the k_d selected

    precision_d = h_d / k_d
    recall_d    = h_d / m_d          only if m_d > 0, otherwise UNDEFINED (excluded, never 0)

    micro precision = sum(h_d) / sum(k_d)
    micro recall    = sum(h_d) / sum(m_d)
    macro precision = mean_d precision_d                  (all days)
    macro recall    = mean_d recall_d over days with m_d > 0

Ties: orders are ranked by score descending. If the K-th place falls inside a
group of equal scores, h_d is the EXPECTED value under uniformly random
tie-breaking:  h_d = positives strictly above + free_slots * positives_in_tie / tie_size.
This makes the result independent of row order; a constant score therefore
gives exactly the random-selection expectation.

Random baseline (uniform sampling without replacement inside each day, same k_d):
    E[h_d] = k_d * m_d / n_d,   E[precision_d] = m_d / n_d,   E[recall_d] = k_d / n_d (m_d > 0)
Aggregates of the random baseline use the same formulas/weights as the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


# ----------------------------------------------------------------------------- per-day table

def _day_h(scores: np.ndarray, y: np.ndarray, k: int) -> tuple[float, int]:
    """Expected hits among top-k with random tie-breaking; returns (h, tie_size_at_boundary)."""
    n = len(scores)
    k = min(k, n)
    if k == 0:
        return 0.0, 0
    order = np.argsort(-scores, kind="stable")
    s_sorted = scores[order]
    y_sorted = y[order]
    boundary = s_sorted[k - 1]
    above = s_sorted > boundary
    tie = s_sorted == boundary
    n_above = int(above.sum())
    pos_above = float(y_sorted[above].sum())
    n_tie = int(tie.sum())
    pos_tie = float(y_sorted[tie].sum())
    free = k - n_above
    return pos_above + free * pos_tie / n_tie, (n_tie if n_tie > 1 and free < n_tie else 0)


def daily_table(df: pd.DataFrame, score_col: str, k: int, y_col: str = "is_late", day_col: str = "day") -> pd.DataFrame:
    """One row per day: n, m, k_d, h (expected under random tie-breaking), random expectation."""
    rows = []
    for day, g in df.groupby(day_col, sort=True):
        s = g[score_col].to_numpy(float)
        y = g[y_col].to_numpy(int)
        h, ties = _day_h(s, y, k)
        n, m = len(g), int(y.sum())
        kd = min(k, n)
        rows.append({"day": day, "n": n, "m": m, "k_d": kd, "h": h, "h_random": kd * m / n if n else 0.0,
                     "boundary_tie_size": ties})
    return pd.DataFrame(rows)


def aggregate(t: pd.DataFrame, h_col: str = "h") -> dict:
    """Micro/macro aggregates of a daily table (h_col = 'h' for the model, 'h_random' for random)."""
    if t.empty:
        return {"days": 0}
    h, k, m, n = t[h_col], t["k_d"], t["m"], t["n"]
    pos_days = t[m > 0]
    return {
        "days": int(len(t)),
        "days_without_late": int((m == 0).sum()),
        "orders": int(n.sum()),
        "late": int(m.sum()),
        "reviewed": int(k.sum()),
        "found_late": float(h.sum()),
        "review_share": float(k.sum() / n.sum()),
        "reviewed_per_day": float(k.mean()),
        "found_per_day": float(h.mean()),
        "micro_precision": float(h.sum() / k.sum()),
        "micro_recall": float(h.sum() / m.sum()) if m.sum() else None,
        "macro_precision": float((h / k).mean()),
        "macro_recall": float((pos_days[h_col] / pos_days["m"]).mean()) if len(pos_days) else None,
    }


def summarize(t: pd.DataFrame, K: int) -> dict:
    """Model aggregates, analytic random expectation and lifts with identical definitions."""
    model, rnd = aggregate(t, "h"), aggregate(t, "h_random")
    model["days_n_lt_K"] = rnd["days_n_lt_K"] = int((t["n"] < K).sum())
    out = {"K": K, **{f"model_{k}": v for k, v in model.items()},
           **{f"random_{k}": v for k, v in rnd.items() if k in
              ("found_late", "found_per_day", "micro_precision", "micro_recall", "macro_precision", "macro_recall")}}
    for metric in ("micro_precision", "micro_recall", "macro_precision", "macro_recall", "found_late"):
        mv, rv = model.get(metric), rnd.get(metric)
        out[f"lift_{metric}"] = (mv / rv) if (mv is not None and rv) else None
    out["boundary_tie_days"] = int((t["boundary_tie_size"] > 0).sum())
    return out


# ----------------------------------------------------------------------------- random Monte Carlo

def random_monte_carlo(df: pd.DataFrame, k: int, reps: int = 500, seed: int = 0,
                       y_col: str = "is_late", day_col: str = "day") -> pd.DataFrame:
    """Uniform sampling WITHOUT replacement of k_d orders inside each day, `reps` times.

    Spread = variability of random selection on this FIXED set of days, not
    uncertainty about future periods.
    """
    rng = np.random.default_rng(seed)
    days = [g[y_col].to_numpy(int) for _, g in df.groupby(day_col, sort=True)]
    out = []
    for r in range(reps):
        rows = []
        for y in days:
            n = len(y)
            kd = min(k, n)
            idx = rng.choice(n, size=kd, replace=False)
            rows.append((n, int(y.sum()), kd, float(y[idx].sum())))
        t = pd.DataFrame(rows, columns=["n", "m", "k_d", "h"])
        a = aggregate(t, "h")
        out.append({"rep": r, **{k_: a[k_] for k_ in ("micro_precision", "micro_recall", "macro_precision",
                                                       "macro_recall", "found_late")}})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- bootstrap

def block_ids(days: pd.Series, block: str) -> np.ndarray:
    d = pd.to_datetime(pd.Series(days))
    if block == "day":
        return d.dt.strftime("%Y-%m-%d").to_numpy()
    if block == "week":
        return (d - pd.to_timedelta(d.dt.dayofweek, unit="D")).dt.strftime("%Y-%m-%d").to_numpy()
    raise ValueError(block)


def paired_bootstrap(tables: dict[str, pd.DataFrame], pair: tuple[str, str], reps: int, seed: int,
                     block: str = "day") -> pd.DataFrame:
    """Paired bootstrap over days (or consecutive ISO weeks) of daily top-K tables.

    The SAME resampled blocks are used for every model, so the difference
    pair[0] - pair[1] is estimated on identical days and orders.
    """
    names = list(tables)
    base = tables[names[0]]
    for t in tables.values():
        if list(t["day"]) != list(base["day"]):
            raise ValueError("daily tables must cover identical days")
    blocks = block_ids(base["day"], block)
    uniq = np.unique(blocks)
    members = {b: np.flatnonzero(blocks == b) for b in uniq}
    rng = np.random.default_rng(seed)
    arr = {n: (t["h"].to_numpy(float), t["k_d"].to_numpy(float), t["m"].to_numpy(float)) for n, t in tables.items()}
    rows = []
    for r in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[b] for b in pick])
        row = {"rep": r}
        for n, (h, k, m) in arr.items():
            hh, kk, mm = h[idx], k[idx], m[idx]
            row[f"{n}_micro_precision"] = hh.sum() / kk.sum()
            row[f"{n}_found_per_day"] = hh.mean()
            row[f"{n}_micro_recall"] = hh.sum() / mm.sum() if mm.sum() else np.nan
        a, b = pair
        for metric in ("micro_precision", "found_per_day", "micro_recall"):
            row[f"diff_{metric}"] = row[f"{a}_{metric}"] - row[f"{b}_{metric}"]
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_ap(df: pd.DataFrame, score_cols: dict[str, str], pair: tuple[str, str], reps: int, seed: int,
                        block: str = "day", y_col: str = "is_late", day_col: str = "day") -> pd.DataFrame:
    """Paired block bootstrap of Average Precision (order-level, blocks = days or weeks)."""
    blocks = block_ids(df[day_col], block)
    uniq = np.unique(blocks)
    members = {b: np.flatnonzero(blocks == b) for b in uniq}
    y = df[y_col].to_numpy(int)
    scores = {n: df[c].to_numpy(float) for n, c in score_cols.items()}
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[b] for b in pick])
        row = {"rep": r}
        for n, s in scores.items():
            row[f"{n}_ap"] = average_precision_score(y[idx], s[idx]) if y[idx].any() else np.nan
        row["diff_ap"] = row[f"{pair[0]}_ap"] - row[f"{pair[1]}_ap"]
        rows.append(row)
    return pd.DataFrame(rows)


def interval(x: pd.Series, level: float = 0.95) -> dict:
    x = pd.Series(x).dropna()
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return {"mean": float(x.mean()), "ci_low": float(x.quantile(lo)), "ci_high": float(x.quantile(hi)),
            "share_positive": float((x > 0).mean())}
