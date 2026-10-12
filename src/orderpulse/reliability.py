"""Reliability & queue evaluation stage (outputs in experiments/reliability_v2).

Parts:
  1. MVP v1 on the ORIGINAL test (already viewed): fixed predictions only --
     daily top-K, analytic + Monte Carlo random baseline, paired bootstrap
     CatBoost vs LogReg, monthly/weekly stability. No selection happens here.
  2. Development backtests W1..W3 before the original test: calendar ablation
     A/B/C with fixed parameters, K-curves, paired bootstrap of differences.
  3. Calibration feasibility: model -> sigmoid calibrator -> later check window.

The production model and predictions in artifacts/ are never written here.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Config
from .features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from .queue_eval import (block_ids, daily_table, interval, paired_bootstrap, paired_bootstrap_ap,
                         random_monte_carlo, summarize)
from .splits import assert_labels_available, make_splits, periods
from .train import load_ml_frame
from .utils import LOG, timed, write_json_atomic

EXP_DIR = Path("experiments") / "reliability_v2"
MODELS = ["catboost", "logreg", "rule_short_promise", "constant"]


def load_reliability_config(cfg: Config) -> dict:
    with open(cfg.root / "configs" / "reliability.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================================ helpers

def _ts(x) -> pd.Timestamp:
    return pd.Timestamp(x)


def _in(df: pd.DataFrame, start, end) -> pd.Series:
    return (df["prediction_time"] >= _ts(start)) & (df["prediction_time"] < _ts(end))


def _X(df: pd.DataFrame, num: list[str], cat: list[str]) -> pd.DataFrame:
    X = df.loc[:, num + cat].copy()
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    for c in cat:
        X[c] = X[c].astype("object").where(X[c].notna(), "unknown").astype(str)
    return X


def fit_catboost(tr, es, num, cat, params, seed):
    p = dict(params)
    model = CatBoostClassifier(random_seed=seed, verbose=False, allow_writing_files=False, thread_count=-1, **p)
    model.fit(Pool(_X(tr, num, cat), tr["is_late"].to_numpy(), cat_features=cat),
              eval_set=Pool(_X(es, num, cat), es["is_late"].to_numpy(), cat_features=cat), use_best_model=True)
    return model


def fit_logreg(tr, num, cat, seed):
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), cat),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=seed))])
    pipe.fit(_X(tr, num, cat), tr["is_late"].to_numpy())
    return pipe


def topk_rows(df: pd.DataFrame, models: list[str], k_list: list[int], tags: dict) -> tuple[list[dict], list[pd.DataFrame]]:
    rows, dailies = [], []
    for m in models:
        for k in k_list:
            t = daily_table(df, m, k)
            rows.append({**tags, "model": m, **summarize(t, k)})
            dailies.append(t.assign(**tags, model=m, K=k))
    return rows, dailies


def ranking(df: pd.DataFrame, col: str) -> dict:
    y = df["is_late"].to_numpy()
    s = df[col].to_numpy(float)
    both = len(np.unique(y)) == 2
    return {"n": int(len(y)), "late": int(y.sum()), "late_rate": float(y.mean()),
            "ap": float(average_precision_score(y, s)) if both else None,
            "roc_auc": float(roc_auc_score(y, s)) if both else None}


def _boot_summary(bs: pd.DataFrame, cols: list[str], tags: dict) -> list[dict]:
    return [{**tags, "statistic": c, **interval(bs[c])} for c in cols]


# ============================================================================ part 1: v1 on viewed test

def v1_test_frame(cfg: Config, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds, _ = make_splits(frame, periods(cfg.model))
    preds = pd.read_parquet(cfg.artifacts_dir / "predictions.parquet")
    wide = preds.pivot_table(index="order_id", columns="model_name", values="risk_score")
    test = ds["test"].join(wide, on="order_id")
    test["day"] = test["prediction_time"].dt.date
    # sensitivity: test orders delivered after data_cutoff (excluded by the label rule)
    p_test = [p for p in periods(cfg.model) if p.name == "test"][0]
    extra = frame[_in(frame, p_test.start, p_test.end) & frame["ml_eligible"]
                  & (frame["label_available_at"] >= p_test.label_available_before)].join(wide, on="order_id")
    extra = extra.assign(day=extra["prediction_time"].dt.date, is_late=extra["is_late"].astype(int))
    return test, extra


def part1_v1(cfg: Config, rc: dict, frame: pd.DataFrame, out: Path, log: list) -> dict:
    t0 = time.perf_counter()
    test, extra = v1_test_frame(cfg, frame)
    k_list = rc["k_list"]
    rows, dailies = topk_rows(test, MODELS, k_list, {"period": "test_v1_viewed"})
    topk = pd.DataFrame(rows)
    topk.to_csv(out / "topk_test_v1.csv", index=False)
    pd.concat(dailies).to_csv(out / "daily_test_v1.csv", index=False)

    mc_rows = []
    for k in k_list:
        mc = random_monte_carlo(test, k, rc["random_mc"]["reps"], rc["random_mc"]["seed"] + k)
        for c in ["micro_precision", "micro_recall", "macro_precision", "macro_recall", "found_late"]:
            x = mc[c].dropna()
            mc_rows.append({"period": "test_v1_viewed", "K": k, "metric": c, "mc_mean": float(x.mean()),
                            "mc_sd": float(x.std()), "mc_p2_5": float(x.quantile(0.025)),
                            "mc_p97_5": float(x.quantile(0.975)), "reps": len(x)})
    mc_df = pd.DataFrame(mc_rows)
    mc_df.to_csv(out / "random_mc_test_v1.csv", index=False)

    b = rc["bootstrap"]
    boot = []
    for block in b["blocks"]:
        for k in b["k_list"]:
            tabs = {m: daily_table(test, m, k) for m in ("catboost", "logreg")}
            bs = paired_bootstrap(tabs, ("catboost", "logreg"), b["reps"], b["seed"] + k, block)
            boot += _boot_summary(bs, ["catboost_micro_precision", "logreg_micro_precision", "diff_micro_precision",
                                       "catboost_found_per_day", "logreg_found_per_day", "diff_found_per_day"],
                                  {"period": "test_v1_viewed", "block": block, "K": k})
        bs = paired_bootstrap_ap(test, {"catboost": "catboost", "logreg": "logreg"}, ("catboost", "logreg"),
                                 b["reps"], b["seed"], block)
        boot += _boot_summary(bs, ["catboost_ap", "logreg_ap", "diff_ap"],
                              {"period": "test_v1_viewed", "block": block, "K": None})
    boot_df = pd.DataFrame(boot)
    boot_df.to_csv(out / "bootstrap_test_v1.csv", index=False)

    # stability by month / ISO week
    stab = []
    for unit in ("month", "week"):
        key = (test["prediction_time"].dt.strftime("%Y-%m") if unit == "month"
               else pd.Series(block_ids(test["day"], "week"), index=test.index))
        for g, part in test.groupby(key):
            row = {"unit": unit, "period": g, "days": int(part["day"].nunique()), "orders": int(len(part)),
                   "late": int(part["is_late"].sum()), "late_rate": float(part["is_late"].mean())}
            for m in ("catboost", "logreg"):
                r = ranking(part, m)
                s = summarize(daily_table(part, m, 50), 50)
                row.update({f"{m}_ap": r["ap"], f"{m}_p50_micro": s["model_micro_precision"],
                            f"{m}_found_per_day_k50": s["model_found_per_day"]})
            row["random_p50_micro"] = s["random_micro_precision"]
            row["lift_catboost_p50"] = row["catboost_p50_micro"] / row["random_p50_micro"] if row["random_p50_micro"] else None
            stab.append(row)
    stab_df = pd.DataFrame(stab)
    stab_df.to_csv(out / "stability_test_v1.csv", index=False)

    # sensitivity: add back the test orders delivered after data_cutoff
    sens = {"excluded_post_cutoff_orders": int(len(extra)), "excluded_post_cutoff_late": int(extra["is_late"].sum())}
    if len(extra):
        both = pd.concat([test, extra[test.columns]], ignore_index=True)
        for m in ("catboost", "logreg"):
            sens[f"{m}_ap_with"] = ranking(both, m)["ap"]
            sens[f"{m}_ap_without"] = ranking(test, m)["ap"]
            sens[f"{m}_p50_with"] = summarize(daily_table(both, m, 50), 50)["model_micro_precision"]
            sens[f"{m}_p50_without"] = summarize(daily_table(test, m, 50), 50)["model_micro_precision"]
    rank_rows = [{"model": m, **ranking(test, m)} for m in MODELS]
    log.append({"experiment": "v1_viewed_test_audit", "seconds": round(time.perf_counter() - t0, 1),
                "rows": int(len(test)), "days": int(test["day"].nunique())})
    return {"topk": topk, "mc": mc_df, "bootstrap": boot_df, "stability": stab_df, "sensitivity": sens,
            "ranking": rank_rows, "n": int(len(test)), "days": int(test["day"].nunique())}


# ============================================================================ part 2: development backtests

def window_sets(frame: pd.DataFrame, w: dict, cutoff: pd.Timestamp) -> tuple[dict, dict]:
    elig = frame[frame["ml_eligible"]].copy()
    elig["is_late"] = elig["is_late"].astype(int)
    sets, checks = {}, {}
    for part in ("fit", "es"):
        spec = w[part]
        d = elig[_in(elig, spec["start"], spec["end"])]
        avail = d["label_available_at"] < _ts(spec["label_cutoff"])
        sets[part] = d[avail].reset_index(drop=True)
        checks[part] = {**assert_labels_available(sets[part], spec["label_cutoff"], part),
                        "excluded_label_not_yet_available": int((~avail).sum()), "late": int(sets[part]["is_late"].sum())}
    spec = w["eval"]
    d = elig[_in(elig, spec["start"], spec["end"])]
    avail = d["label_available_at"] < cutoff
    sets["eval"] = d[avail].reset_index(drop=True)
    strict = d["label_available_at"] < _ts("2018-04-01")
    checks["eval"] = {"rows": int(len(sets["eval"])), "late": int(sets["eval"]["is_late"].sum()),
                      "label_rule": f"matured: delivered before data_cutoff {cutoff.date()}",
                      "would_be_excluded_if_labels_before_2018_04_01": int((avail & ~strict).sum())}
    sets["eval"]["day"] = sets["eval"]["prediction_time"].dt.date
    return sets, checks


def part2_backtests(cfg: Config, rc: dict, frame: pd.DataFrame, out: Path, log: list) -> dict:
    cutoff = _ts(cfg.data["data_cutoff"])
    seed = int(rc["seed"])
    preds_dir = out / "dev_predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    abl_rows, topk_all, daily_all, window_info, evals = [], [], [], {}, {}
    for wname, w in rc["windows"].items():
        sets, checks = window_sets(frame, w, cutoff)
        window_info[wname] = {"spec": w, "checks": checks}
        ev = sets["eval"].copy()
        ev["rule_short_promise"] = -pd.to_numeric(ev["promised_days"], errors="coerce").fillna(
            float(sets["fit"]["promised_days"].median()))
        ev["constant"] = float(sets["fit"]["is_late"].mean())
        for vname, drop in rc["variants"].items():
            num = [c for c in NUMERIC_FEATURES if c not in drop]
            cat = [c for c in CATEGORICAL_FEATURES if c not in drop]
            t0 = time.perf_counter()
            cb = fit_catboost(sets["fit"], sets["es"], num, cat, rc["catboost_params"], seed)
            t_cb = time.perf_counter() - t0
            t0 = time.perf_counter()
            lr = fit_logreg(sets["fit"], num, cat, seed)
            t_lr = time.perf_counter() - t0
            ev[f"catboost_{vname}"] = cb.predict_proba(Pool(_X(ev, num, cat), cat_features=cat))[:, 1]
            ev[f"logreg_{vname}"] = lr.predict_proba(_X(ev, num, cat))[:, 1]
            for m, secs, extra in [("catboost", t_cb, {"best_iteration": int(cb.get_best_iteration()),
                                                       "tree_count": int(cb.tree_count_)}),
                                   ("logreg", t_lr, {})]:
                col = f"{m}_{vname}"
                r = ranking(ev, col)
                row = {"window": wname, "variant": vname, "model": m, "n_features": len(num) + len(cat),
                       "fit_seconds": round(secs, 2), **extra, **r}
                for k in (10, 20, 50):
                    s = summarize(daily_table(ev, col, k), k)
                    row[f"p{k}_micro"] = s["model_micro_precision"]
                    row[f"random_p{k}_micro"] = s["random_micro_precision"]
                    row[f"lift_p{k}"] = s["lift_micro_precision"]
                abl_rows.append(row)
                log.append({"experiment": f"dev_{wname}_{vname}_{m}", "seconds": round(secs, 2),
                            "fit_rows": int(len(sets["fit"])), "es_rows": int(len(sets["es"])),
                            "eval_rows": int(len(ev))})
        for m in ("rule_short_promise", "constant"):
            r = ranking(ev, m)
            row = {"window": wname, "variant": "-", "model": m, "n_features": 1 if m != "constant" else 0, **r}
            for k in (10, 20, 50):
                s = summarize(daily_table(ev, m, k), k)
                row[f"p{k}_micro"] = s["model_micro_precision"]
                row[f"random_p{k}_micro"] = s["random_micro_precision"]
                row[f"lift_p{k}"] = s["lift_micro_precision"]
            abl_rows.append(row)
        # K-curves on development eval windows: production-equivalent variant A + baselines
        cols = {"catboost": "catboost_A", "logreg": "logreg_A", "rule_short_promise": "rule_short_promise",
                "constant": "constant"}
        for m, c in cols.items():
            for k in rc["k_list"]:
                t = daily_table(ev, c, k)
                topk_all.append({"period": f"dev_{wname}", "model": m, **summarize(t, k)})
                daily_all.append(t.assign(period=f"dev_{wname}", model=m, K=k))
        keep = ["order_id", "prediction_time", "day", "is_late"] + [c for c in ev.columns if c.startswith(
            ("catboost_", "logreg_"))] + ["rule_short_promise", "constant"]
        ev[keep].to_parquet(preds_dir / f"{wname}_eval.parquet", index=False)
        evals[wname] = ev[keep].assign(window=wname)

    pooled = pd.concat(evals.values(), ignore_index=True)
    for m, c in {"catboost": "catboost_A", "logreg": "logreg_A", "rule_short_promise": "rule_short_promise",
                 "constant": "constant"}.items():
        for k in rc["k_list"]:
            t = daily_table(pooled, c, k)
            topk_all.append({"period": "dev_pooled", "model": m, **summarize(t, k)})
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(out / "ablation_dev.csv", index=False)
    topk = pd.DataFrame(topk_all)
    topk.to_csv(out / "topk_dev.csv", index=False)
    pd.concat(daily_all).to_csv(out / "daily_dev.csv", index=False)

    # paired bootstrap on pooled development days
    b = rc["bootstrap"]
    boot = []
    pairs = [("catboost_A", "logreg_A"), ("catboost_A", "catboost_B"), ("catboost_A", "catboost_C"),
             ("logreg_A", "logreg_C")]
    for block in b["blocks"]:
        for a, c in pairs:
            for k in b["k_list"]:
                tabs = {a: daily_table(pooled, a, k), c: daily_table(pooled, c, k)}
                bs = paired_bootstrap(tabs, (a, c), b["reps"], b["seed"] + k, block)
                boot += _boot_summary(bs, ["diff_micro_precision", "diff_found_per_day"],
                                      {"period": "dev_pooled", "block": block, "K": k, "pair": f"{a} - {c}"})
            bs = paired_bootstrap_ap(pooled, {a: a, c: c}, (a, c), b["reps"], b["seed"], block)
            boot += _boot_summary(bs, ["diff_ap"], {"period": "dev_pooled", "block": block, "K": None,
                                                    "pair": f"{a} - {c}"})
    boot_df = pd.DataFrame(boot)
    boot_df.to_csv(out / "bootstrap_dev.csv", index=False)
    return {"ablation": abl, "topk": topk, "bootstrap": boot_df, "windows": window_info, "pooled": pooled}


# ============================================================================ part 3: calibration feasibility

def part3_calibration(cfg: Config, rc: dict, frame: pd.DataFrame, out: Path, log: list) -> dict:
    c = rc["calibration"]
    w = rc["windows"][c["model_window"]]
    cutoff = _ts(cfg.data["data_cutoff"])
    sets, _ = window_sets(frame, w, cutoff)
    model = fit_catboost(sets["fit"], sets["es"], NUMERIC_FEATURES, CATEGORICAL_FEATURES, rc["catboost_params"],
                         int(rc["seed"]))
    elig = frame[frame["ml_eligible"]].copy()
    elig["is_late"] = elig["is_late"].astype(int)
    cal = elig[_in(elig, c["calib"]["start"], c["calib"]["end"])]
    cal_avail = cal["label_available_at"] < _ts(c["calib"]["label_cutoff"])
    cal_used = cal[cal_avail].reset_index(drop=True)
    check_cal = assert_labels_available(cal_used, c["calib"]["label_cutoff"], "calibrator fit")
    chk = elig[_in(elig, c["check"]["start"], c["check"]["end"]) & (elig["label_available_at"] < cutoff)].reset_index(drop=True)
    if not (_ts(w["es"]["label_cutoff"]) <= _ts(c["calib"]["start"]) and _ts(c["calib"]["label_cutoff"]) <= _ts(c["check"]["start"])):
        raise ValueError("calibration windows are not sequential")

    def raw(df):
        p = model.predict_proba(Pool(_X(df, NUMERIC_FEATURES, CATEGORICAL_FEATURES), cat_features=CATEGORICAL_FEATURES))[:, 1]
        return np.clip(p, 1e-6, 1 - 1e-6)

    p_cal, p_chk = raw(cal_used), raw(chk)
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_cal).reshape(-1, 1), cal_used["is_late"])
    p_chk_cal = platt.predict_proba(logit(p_chk).reshape(-1, 1))[:, 1]
    y = chk["is_late"].to_numpy()
    # transparent bins: deciles of the UNCALIBRATED check-window score
    edges = np.quantile(p_chk, np.linspace(0, 1, c["n_bins"] + 1))
    bins = np.clip(np.searchsorted(edges, p_chk, side="right") - 1, 0, c["n_bins"] - 1)
    rel = []
    for bi in range(c["n_bins"]):
        mask = bins == bi
        rel.append({"bin": bi, "n": int(mask.sum()), "late": int(y[mask].sum()),
                    "observed": float(y[mask].mean()) if mask.any() else None,
                    "mean_uncalibrated": float(p_chk[mask].mean()) if mask.any() else None,
                    "mean_calibrated": float(p_chk_cal[mask].mean()) if mask.any() else None})

    def scores(p):
        return {"brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, p)),
                "mean_prediction": float(p.mean()), "observed_rate": float(y.mean())}

    res = {
        "scheme": f"model fitted in {c['model_window']} (fit {w['fit']['start']}..{w['fit']['end']}, ES {w['es']['start']}.."
                  f"{w['es']['end']}) -> sigmoid calibrator on approvals {c['calib']['start']}..{c['calib']['end']} with labels "
                  f"observed before {c['calib']['label_cutoff']} -> check on approvals {c['check']['start']}..{c['check']['end']}",
        "calibrator": {"a": float(platt.coef_[0][0]), "b": float(platt.intercept_[0])},
        "calib_window": {**check_cal, "late": int(cal_used["is_late"].sum()), "late_rate": float(cal_used["is_late"].mean()),
                         "excluded_label_not_yet_available": int((~cal_avail).sum()),
                         "late_rate_with_matured_labels": float(cal[cal["label_available_at"] < cutoff]["is_late"].mean())},
        "check_window": {"rows": int(len(chk)), "late": int(y.sum())},
        "uncalibrated": scores(p_chk), "calibrated": scores(p_chk_cal),
        "constant_at_calib_rate": scores(np.full(len(y), cal_used["is_late"].mean())),
        "reliability_bins": rel,
    }
    write_json_atomic(res, out / "calibration.json")
    log.append({"experiment": "calibration_feasibility", "calib_rows": int(len(cal_used)), "check_rows": int(len(chk))})
    return res


# ============================================================================ figures

def figures(out: Path, v1_topk: pd.DataFrame, dev_topk: pd.DataFrame, cal: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"catboost": "#2a78d6", "logreg": "#eb6834", "rule_short_promise": "#1baf7a", "random": "#8a8985"}
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    panels = [("model_micro_precision", "random_micro_precision", "Precision@K (micro)", "precision_vs_k"),
              ("model_micro_recall", "random_micro_recall", "Recall@K (micro)", "recall_vs_k"),
              ("lift_micro_precision", None, "Lift precision vs random expectation", "lift_vs_k")]
    for period_name, df in [("dev_pooled", dev_topk[dev_topk["period"] == "dev_pooled"]),
                            ("test_v1_viewed", v1_topk)]:
        for col, rcol, title, fname in panels:
            fig, ax = plt.subplots(figsize=(6, 4))
            for m in ("catboost", "logreg", "rule_short_promise"):
                d = df[df["model"] == m].sort_values("K")
                ax.plot(d["K"], d[col], marker="o", color=colors[m], label=m, lw=2)
            if rcol:
                d = df[df["model"] == "catboost"].sort_values("K")
                ax.plot(d["K"], d[rcol], ls="--", color=colors["random"], label="random (expected)", lw=1.5)
            else:
                ax.axhline(1, ls="--", color=colors["random"], lw=1)
            ax.set_xscale("log")
            ax.set_xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
            ax.set_xlabel("K per approval day")
            ax.set_title(f"{title} — {period_name}")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(figdir / f"{fname}_{period_name}.png", dpi=110)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in ("catboost", "logreg", "rule_short_promise"):
            d = df[df["model"] == m].sort_values("K")
            ax.plot(d["model_reviewed_per_day"], d["model_found_per_day"], marker="o", color=colors[m], label=m, lw=2)
        d = df[df["model"] == "catboost"].sort_values("K")
        ax.plot(d["model_reviewed_per_day"], d["random_found_per_day"], ls="--", color=colors["random"],
                label="random (expected)")
        ax.set_xlabel("orders reviewed per day")
        ax.set_ylabel("late orders found per day")
        ax.set_title(f"Found late vs reviews — {period_name}")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / f"found_vs_reviewed_{period_name}.png", dpi=110)
        plt.close(fig)
    rel = pd.DataFrame(cal["reliability_bins"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    mx = float(max(rel["observed"].max(), rel["mean_calibrated"].max(), rel["mean_uncalibrated"].max()))
    ax.plot([0, mx], [0, mx], ls=":", color=colors["random"])
    ax.plot(rel["mean_uncalibrated"], rel["observed"], marker="o", color=colors["catboost"], label="uncalibrated")
    ax.plot(rel["mean_calibrated"], rel["observed"], marker="s", color=colors["logreg"], label="sigmoid-calibrated")
    ax.set_xlabel("mean predicted")
    ax.set_ylabel("observed late rate")
    ax.set_title("Reliability, check window (deciles of raw score)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "calibration_reliability.png", dpi=110)
    plt.close(fig)


# ============================================================================ entry point

def run_reliability(cfg: Config) -> dict:
    with timed("reliability"):
        rc = load_reliability_config(cfg)
        out = cfg.root / EXP_DIR
        out.mkdir(parents=True, exist_ok=True)
        frame, fp = load_ml_frame(cfg)
        meta = json.loads((cfg.artifacts_dir / "model_metadata.json").read_text(encoding="utf-8"))
        if fp != meta["data_fingerprint"]:
            raise RuntimeError("warehouse fingerprint differs from MVP v1 model")
        log: list[dict] = []
        with timed("reliability:v1_test"):
            v1 = part1_v1(cfg, rc, frame, out, log)
        with timed("reliability:dev_backtests"):
            dev = part2_backtests(cfg, rc, frame, out, log)
        with timed("reliability:calibration"):
            cal = part3_calibration(cfg, rc, frame, out, log)
        figures(out, v1["topk"], dev["topk"], cal)
        write_json_atomic({
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_fingerprint": fp,
            "mvp_v1_model_versions": {n: m["version"] for n, m in meta["models"].items()},
            "reliability_config": rc,
            "test_status": "original test (approvals 2018-04-01..2018-06-30) already viewed; used only for "
                           "descriptive audit of fixed MVP v1 predictions",
            "dev_windows": dev["windows"],
            "v1_test": {"rows": v1["n"], "days": v1["days"], "ranking": v1["ranking"],
                        "post_cutoff_sensitivity": v1["sensitivity"]},
        }, out / "config.json")
        write_json_atomic(log, out / "experiments_log.json")
        return {"v1": v1, "dev": dev, "calibration": cal}
