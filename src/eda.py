"""Exploratory figures. Descriptive only; the test window is not used to pick tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FIGURES, OUTPUTS
from src.data import load_analysis
from src.io_utils import write_json
from src.plots import ACCENT, ACCENT_2, MUTED, apply_style, finish

import matplotlib.pyplot as plt


def _rate(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def run_eda(df: pd.DataFrame | None = None) -> dict:
    apply_style()
    if df is None:
        df, _ = load_analysis()
    primary = df.loc[df["in_primary"]].copy()
    FIGURES.mkdir(parents=True, exist_ok=True)
    return _run(df, primary)


def _run(df: pd.DataFrame, primary: pd.DataFrame) -> dict:
    from src.io_utils import read_json

    build = read_json(OUTPUTS / "build_summary.json")
    n_parsed = int(build["n_questions_parsed"])

    n_saved = int(len(df))
    n_primary = int(len(primary))
    n_answered = int(primary["answered_24h"].sum())
    n_eventual = int(primary["eventually_answered"].sum())

    funnel_labels = [
        "Parsed questions",
        "Still observable\n(not in the last 24h)",
        "Primary set\n(not closed unanswered)",
        "Answered by someone\nelse within 24 hours",
        "Answered by someone\nelse eventually",
    ]
    funnel_vals = [n_parsed, n_saved, n_primary, n_answered, n_eventual]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(funnel_vals))[::-1]
    ax.barh(y, funnel_vals, color=ACCENT)
    ax.set_yticks(y, funnel_labels)
    ax.set_xlabel("Questions")
    ax.set_title("Most questions are still open long enough to score")
    for yi, val in zip(y, funnel_vals):
        ax.text(val, yi, f"  {val:,.0f}", va="center", color=MUTED, fontsize=10)
    ax.set_xlim(0, max(funnel_vals) * 1.18)
    finish(fig, FIGURES / "01_funnel.png")

    by_year = primary.groupby("year")["answered_24h"].agg(rate="mean", n="size").reset_index()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(by_year["year"], by_year["rate"], color=ACCENT, marker="o")
    ax.set_ylim(0, 1)
    ax.set_ylabel("24-hour answer rate")
    ax.set_xlabel("Year posted")
    ax.set_title("Answer rate by year, primary set")
    finish(fig, FIGURES / "02_answer_rate_by_year.png")

    by_tag = (
        primary.groupby("tag_family")["answered_24h"]
        .agg(rate="mean", n="size")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    order = by_tag.sort_values("rate")
    ax.barh(order["tag_family"], order["rate"], color=ACCENT)
    ax.set_xlim(0, 1)
    ax.set_xlabel("24-hour answer rate")
    ax.set_title("Answer rate by tag family (families learned on the train window)")
    finish(fig, FIGURES / "03_tag_family_rates.png")

    weekend = (
        primary.groupby("weekend")["answered_24h"].mean().rename({False: "weekday", True: "weekend"})
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    hours = primary.groupby("hour_utc")["answered_24h"].mean()
    ax.plot(hours.index, hours.values, color=ACCENT, marker="o", ms=4)
    ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Hour posted (UTC)")
    ax.set_ylabel("24-hour answer rate")
    ax.set_title("UTC hour mixes regions; there is no user timezone in this dump")
    finish(fig, FIGURES / "04_hour_of_day.png")

    bins = [0, 20, 40, 60, 80, 100, 10_000]
    labels = ["1–20", "21–40", "41–60", "61–80", "81–100", "101+"]
    primary = primary.copy()
    primary["title_bin"] = pd.cut(primary["title_len"].clip(lower=1), bins=bins, labels=labels, right=True)
    title_rates = primary.groupby("title_bin", observed=True)["answered_24h"].agg(rate="mean", n="size")
    se = np.sqrt(title_rates["rate"] * (1 - title_rates["rate"]) / title_rates["n"].clip(lower=1))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(title_rates.index.astype(str), title_rates["rate"], color=ACCENT, yerr=1.96 * se, capsize=3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("24-hour answer rate")
    ax.set_xlabel("Title length (characters)")
    ax.set_title("Raw answer rate by title length")
    finish(fig, FIGURES / "05_title_length.png")

    hours_known = primary.loc[primary["eventually_answered"], "hours_to_other_answer"].dropna()
    hours_known = hours_known.clip(lower=0, upper=24 * 14)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    grid = np.linspace(0, 24 * 7, 200)
    cdf = [(hours_known <= g).mean() for g in grid]
    ax.plot(grid, cdf, color=ACCENT)
    ax.axvline(24, color=ACCENT_2, ls="--", label="24-hour label")
    ax.set_xlabel("Hours until the first answer from someone else")
    ax.set_ylabel("Share of eventually answered questions")
    ax.set_title("A 24-hour cutoff sits on a long tail, not at the median")
    ax.legend()
    finish(fig, FIGURES / "06_time_to_answer.png")

    limits = pd.Series(
        {
            "Null owner id": primary["owner_user_id"].isna().mean(),
            "Edit marker in body": primary["has_edit_marker"].mean(),
            "No code block": (~primary["has_code"]).mean(),
            "Untagged or rare family": (primary["tag_family"] == "other").mean(),
            "Weekend post": primary["weekend"].mean(),
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.barh(limits.index, limits.values, color=ACCENT_2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of the primary set")
    ax.set_title("Limits that change how the label and features should be read")
    finish(fig, FIGURES / "07_data_limits.png")

    payload = {
        "n_parsed": n_parsed,
        "n_observable": n_saved,
        "n_primary": n_primary,
        "n_closed_fast_excluded": int(df["closed_fast_unanswered"].sum()),
        "answer_rate_24h": _rate(primary["answered_24h"]),
        "eventual_answer_rate": _rate(primary["eventually_answered"]),
        "share_of_eventual_within_24h": float(
            (primary.loc[primary["eventually_answered"], "hours_to_other_answer"].clip(lower=0) <= 24).mean()
        ),
        "median_hours_to_answer": float(hours_known.median()) if len(hours_known) else None,
        "weekday_answer_rate": float(weekend.get("weekday", np.nan)),
        "weekend_answer_rate": float(weekend.get("weekend", np.nan)),
        "answer_rate_by_year": {
            str(int(r.year)): {"rate": float(r.rate), "n": int(r.n)} for r in by_year.itertuples(index=False)
        },
        "null_owner_share": float(primary["owner_user_id"].isna().mean()),
        "edit_marker_share": float(primary["has_edit_marker"].mean()),
        "has_code_share": float(primary["has_code"].mean()),
        "other_tag_share": float((primary["tag_family"] == "other").mean()),
    }
    write_json(OUTPUTS / "eda_summary.json", payload)
    print(
        f"Primary n={n_primary:,}  24h answer rate={payload['answer_rate_24h']:.3f}  "
        f"eventual={payload['eventual_answer_rate']:.3f}"
    )
    return payload


if __name__ == "__main__":
    run_eda()
