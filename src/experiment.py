"""Sample-size the ask-wizard experiment. This does not pretend the dump is an experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import (
    EXPERIMENT_ALPHA,
    EXPERIMENT_MDE,
    EXPERIMENT_POWER,
    EXPERIMENT_TAGS,
    GUARDRAIL_RELATIVE_DROP,
    NOVELTY_DAYS,
    OUTPUTS,
    SAMPLE_FRACTION,
)
from src.data import load_analysis
from src.io_utils import write_json


def _icc(df: pd.DataFrame) -> dict:
    """One-way random-effects ICC of the 24-hour answer indicator within user."""
    work = df.loc[df["owner_user_id"].notna(), ["owner_user_id", "answered_24h"]].copy()
    work["answered_24h"] = work["answered_24h"].astype(float)
    stats = work.groupby("owner_user_id")["answered_24h"].agg(count="size", mean="mean", var="var")
    multi = stats.loc[stats["count"] >= 2]
    n = int(multi["count"].sum())
    k = int(len(multi))
    if k < 2 or n <= k:
        return {"icc": 0.0, "clusters": k, "questions": n, "mean_cluster_size": 0.0}
    grand = float(np.average(multi["mean"], weights=multi["count"]))
    ssb = float((multi["count"] * (multi["mean"] - grand) ** 2).sum())
    ssw = float(((multi["count"] - 1) * multi["var"].fillna(0)).sum())
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    m0 = (n - float((multi["count"] ** 2).sum()) / n) / (k - 1)
    if msw == 0:
        icc = 0.0
    else:
        icc = (msb - msw) / (msb + (m0 - 1) * msw)
    icc = float(max(icc, 0.0))
    return {
        "icc": icc,
        "clusters_with_2plus": k,
        "questions_in_those_clusters": n,
        "mean_cluster_size_unbalanced": float(m0),
    }


def _questions_per_user_per_14_days(df: pd.DataFrame) -> float:
    work = df.loc[df["owner_user_id"].notna(), ["owner_user_id", "created"]].copy()
    if work.empty:
        return 1.0
    origin = work["created"].min()
    work["bin"] = ((work["created"] - origin).dt.total_seconds() // (14 * 86400)).astype(int)
    sizes = work.groupby(["owner_user_id", "bin"]).size()
    return float(sizes.mean())


def _sample_size(p1: float, mde: float, alpha: float, power: float) -> float:
    p2 = min(p1 + mde, 0.999)
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(z**2 * (p1 * (1 - p1) + p2 * (1 - p2)) / mde**2)


def run_experiment(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df, _ = load_analysis()
    primary = df.loc[df["in_primary"]].copy()
    train = primary.loc[primary["split"] == "train"]
    recent = primary.loc[primary["split"] == "test"]
    families = (
        train["tag_family"].value_counts().drop(labels=["other"], errors="ignore").head(EXPERIMENT_TAGS).index.tolist()
    )
    scope_recent = recent.loc[recent["tag_family"].isin(families)]
    scope_train = train.loc[train["tag_family"].isin(families)]

    p1 = float(scope_recent["answered_24h"].mean())
    n_per_arm = _sample_size(p1, EXPERIMENT_MDE, EXPERIMENT_ALPHA, EXPERIMENT_POWER)
    icc = _icc(scope_train)
    m_sample = _questions_per_user_per_14_days(scope_recent)
    # The file keeps question ids that are multiples of 10, so a live experiment
    # on every question sees about 10x the volume and about 10x the questions per user.
    m_site = m_sample / SAMPLE_FRACTION
    deff_sample = 1 + (m_sample - 1) * icc["icc"]
    deff_site = 1 + (m_site - 1) * icc["icc"]
    n_site = n_per_arm * deff_site

    span_days = (scope_recent["created"].max() - scope_recent["created"].min()).total_seconds() / 86400
    daily_sample = float(len(scope_recent) / span_days) if span_days else float("nan")
    daily_site = daily_sample / SAMPLE_FRACTION
    days_needed = (2 * n_site) / daily_site if daily_site else float("nan")
    duration = float(max(days_needed, NOVELTY_DAYS))

    payload = {
        "decision": (
            "In the largest tag families, randomize users to a composer that requires a code block "
            "or an explicit 'no code needed' before the question is posted."
        ),
        "unit_of_randomization": "user",
        "why_not_question": "One person posts many questions, so question-level randomization is not independent.",
        "tag_families": families,
        "primary_metric": "24-hour answer rate from someone other than the asker",
        "baseline_answer_rate_test_window": p1,
        "mde": EXPERIMENT_MDE,
        "alpha": EXPERIMENT_ALPHA,
        "power": EXPERIMENT_POWER,
        "questions_per_arm_ignoring_clustering": n_per_arm,
        "icc_train_window": icc,
        "questions_per_user_per_14_days_in_sample": m_sample,
        "questions_per_user_per_14_days_scaled_to_full_site": m_site,
        "design_effect_used": deff_site,
        "design_effect_if_run_on_the_sample_only": deff_sample,
        "questions_per_arm_with_design_effect": n_site,
        "daily_questions_in_sample": daily_sample,
        "daily_questions_scaled_to_full_site": daily_site,
        "days_to_reach_sample_size": days_needed,
        "recommended_duration_days": duration,
        "novelty_floor_days": NOVELTY_DAYS,
        "guardrail": {
            "metric": "questions successfully posted per randomized user",
            "relative_drop_that_fails": GUARDRAIL_RELATIVE_DROP,
            "rule": (
                "Do not ship the wizard if the treatment arm posts more than 5% fewer questions per user "
                "and the 95% interval for that drop excludes zero."
            ),
        },
        "assumptions": [
            "The observational code-block gap is not the effect of forcing a code block. The experiment is what would estimate that effect.",
            "Question ids in this file are a 10% sample, so live-site volume and questions per user are scaled by 10.",
            "The ICC measured on the thinned sample is used, unchanged, at the scaled cluster size. That is a conservative planning choice if the true ICC is smaller.",
            "Run at least 14 days even if the sample size arrives sooner, so a first-week novelty effect is not the result.",
        ],
    }
    write_json(OUTPUTS / "experiment.json", payload)
    print(
        f"Baseline answer rate {p1:.3f}; {n_site:,.0f} questions/arm; "
        f"{duration:.1f} days recommended; ICC={icc['icc']:.4f}"
    )
    return payload


if __name__ == "__main__":
    run_experiment()
