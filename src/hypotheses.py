"""Three pre-specified tests, estimated on the training window only.

Decision rule: a 2 percentage-point gap is the practical bar. With about a
million rows a tiny gap is "significant," so the interval is compared with
the bar, and Holm-adjusted p-values are reported second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm

from src.config import FIGURES, MIN_STRATUM_N, OUTPUTS, PRACTICAL_BAR
from src.data import load_analysis
from src.io_utils import write_json
from src.plots import ACCENT, ACCENT_2, MUTED, apply_style, finish

import matplotlib.pyplot as plt

Z_95 = 1.959963984540054


def mantel_haenszel_rd(
    df: pd.DataFrame,
    exposure: str,
    outcome: str = "answered_24h",
    strata: list[str] | None = None,
    min_n: int = MIN_STRATUM_N,
) -> dict:
    """Pooled risk difference with Mantel–Haenszel weights.

    Variance is the MH-weighted sum of the stratum two-proportion variances.
    Strata with fewer than `min_n` rows in either arm are dropped so a 0/1
    cell cannot contribute a zero variance.
    """
    cols = [exposure, outcome] if not strata else strata + [exposure, outcome]
    sub = df[cols].dropna().copy()
    sub[exposure] = sub[exposure].astype(int)
    sub[outcome] = sub[outcome].astype(int)

    if not strata:
        return _two_proportion(sub, exposure, outcome) | {"strata": 1, "rows": int(len(sub))}

    grouped = (
        sub.groupby(strata + [exposure], observed=True)[outcome]
        .agg(events="sum", n="size")
        .reset_index()
    )
    exposed = grouped.loc[grouped[exposure] == 1].set_index(strata)
    control = grouped.loc[grouped[exposure] == 0].set_index(strata)
    both = exposed.join(control, lsuffix="_1", rsuffix="_0", how="inner")
    both = both[(both["n_1"] >= min_n) & (both["n_0"] >= min_n)]
    if both.empty:
        raise RuntimeError(f"No strata left for exposure={exposure}")

    p1 = both["events_1"] / both["n_1"]
    p0 = both["events_0"] / both["n_0"]
    weight = both["n_1"] * both["n_0"] / (both["n_1"] + both["n_0"])
    rd_i = p1 - p0
    rd = float(np.average(rd_i, weights=weight))
    var_i = p1 * (1 - p1) / both["n_1"] + p0 * (1 - p0) / both["n_0"]
    var = float(np.sum((weight**2) * var_i) / (np.sum(weight) ** 2))
    return _wrap(rd, var) | {
        "strata": int(len(both)),
        "rows": int(both["n_1"].sum() + both["n_0"].sum()),
    }


def _two_proportion(sub: pd.DataFrame, exposure: str, outcome: str) -> dict:
    arm1 = sub.loc[sub[exposure] == 1, outcome]
    arm0 = sub.loc[sub[exposure] == 0, outcome]
    p1 = float(arm1.mean())
    p0 = float(arm0.mean())
    var = p1 * (1 - p1) / len(arm1) + p0 * (1 - p0) / len(arm0)
    out = _wrap(p1 - p0, var)
    out["n_exposed"] = int(len(arm1))
    out["n_control"] = int(len(arm0))
    out["rate_exposed"] = p1
    out["rate_control"] = p0
    return out


def _wrap(rd: float, var: float) -> dict:
    se = float(np.sqrt(max(var, 0.0)))
    z = rd / se if se > 0 else float("inf")
    p = float(2 * norm.sf(abs(z))) if se > 0 else 0.0
    return {
        "rd": rd,
        "se": se,
        "ci95": [rd - Z_95 * se, rd + Z_95 * se],
        "p_value": p,
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    names = list(p_values)
    raw = np.array([p_values[name] for name in names], dtype=float)
    order = np.argsort(raw)
    adj = np.empty(len(raw))
    running = 0.0
    m = len(raw)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * raw[idx])
        value = max(value, running)
        adj[idx] = value
        running = value
    return {name: float(adj[i]) for i, name in enumerate(names)}


def _decision(rd: float, ci: list[float], direction: int, bar: float = PRACTICAL_BAR) -> str:
    lo, hi = ci
    if direction > 0:
        if lo > bar:
            return "larger than the 2 point bar"
        if hi < 0:
            return "in the opposite direction"
        if lo > 0:
            return "positive but not clearly past the 2 point bar"
        return "not distinguishable from zero at 95%"
    if hi < -bar:
        return "larger than the 2 point bar (negative)"
    if lo > 0:
        return "in the opposite direction"
    if hi < 0:
        return "negative but not clearly past the 2 point bar"
    return "not distinguishable from zero at 95%"


def _title_length_test(train: pd.DataFrame) -> dict:
    work = train[["answered_24h", "title_len", "year"]].dropna().copy()
    work["answered_24h"] = work["answered_24h"].astype(int)
    center = float(work["title_len"].mean())
    work["title_c"] = (work["title_len"] - center) / 100.0
    work["title_c2"] = work["title_c"] ** 2
    bins = [0, 20, 40, 60, 80, 100, 10_000]
    labels = ["1–20", "21–40", "41–60", "61–80", "81–100", "101+"]
    work["title_bin"] = pd.cut(work["title_len"].clip(lower=1), bins=bins, labels=labels)
    linear = smf.logit("answered_24h ~ title_c + C(year)", data=work).fit(disp=False, maxiter=200, method="lbfgs")
    quad = smf.logit("answered_24h ~ title_c + title_c2 + C(year)", data=work).fit(disp=False, maxiter=200, method="lbfgs")
    if not linear.mle_retvals.get("converged", False) or not quad.mle_retvals.get("converged", False):
        raise RuntimeError("Title-length logit did not converge; the likelihood-ratio test would be invalid.")
    lr = float(2 * (quad.llf - linear.llf))
    p_value = float(chi2.sf(lr, 1))

    # Year-standardized rate at each bin: average the bin model's predictions
    # after setting every row to that bin.
    binned = smf.logit("answered_24h ~ C(title_bin) + C(year)", data=work).fit(disp=False, maxiter=200, method="lbfgs")
    if not binned.mle_retvals.get("converged", False):
        raise RuntimeError("Binned title-length logit did not converge.")
    standardized = {}
    base = work.copy()
    for label in labels:
        scenario = base.copy()
        scenario["title_bin"] = pd.Categorical([label] * len(scenario), categories=labels)
        standardized[label] = float(binned.predict(scenario).mean())
    rates = list(standardized.values())
    return {
        "center_title_len": center,
        "linear_title_coef_per_100_chars": float(linear.params["title_c"]),
        "quadratic_coef": float(quad.params["title_c2"]),
        "lr_stat": lr,
        "df": 1,
        "p_value": p_value,
        "year_standardized_bin_rates": standardized,
        "bin_rate_range": float(max(rates) - min(rates)),
        "shape_matches_u": _is_u_shape(standardized, labels),
        "n": int(len(work)),
        "linear_llf": float(linear.llf),
        "quadratic_llf": float(quad.llf),
    }


def _is_u_shape(rates: dict[str, float], labels: list[str]) -> bool:
    values = [rates[label] for label in labels]
    peak = int(np.argmax(values))
    return 0 < peak < len(values) - 1 and values[0] < values[peak] and values[-1] < values[peak]


def _family_plot(train: pd.DataFrame) -> None:
    apply_style()
    rows = []
    for family, part in train.groupby("tag_family"):
        if part["has_code"].nunique() < 2 or len(part) < 500:
            continue
        est = mantel_haenszel_rd(part, "has_code", strata=["year"])
        rows.append({"tag_family": family, **est, "n": int(len(part))})
    frame = pd.DataFrame(rows).sort_values("rd")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    y = np.arange(len(frame))
    ax.errorbar(
        frame["rd"],
        y,
        xerr=[frame["rd"] - frame["ci95"].str[0], frame["ci95"].str[1] - frame["rd"]],
        fmt="o",
        color=ACCENT,
        capsize=3,
    )
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(PRACTICAL_BAR, color=ACCENT_2, ls="--", lw=1, label="2 point bar")
    ax.set_yticks(y, frame["tag_family"])
    ax.set_xlabel("Answer-rate difference, code block minus none (within year)")
    ax.set_title("Code-block gap by tag family, train window (descriptive)")
    ax.legend(loc="lower right")
    finish(fig, FIGURES / "08_code_effect_by_family.png")

    week_rows = []
    for family, part in train.groupby("tag_family"):
        if part["weekend"].nunique() < 2 or len(part) < 500:
            continue
        est = mantel_haenszel_rd(part, "weekend", strata=["year"])
        week_rows.append({"tag_family": family, "rd": est["rd"], "lo": est["ci95"][0], "hi": est["ci95"][1]})
    week = pd.DataFrame(week_rows).sort_values("rd")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    y = np.arange(len(week))
    ax.errorbar(
        week["rd"],
        y,
        xerr=[week["rd"] - week["lo"], week["hi"] - week["rd"]],
        fmt="o",
        color=ACCENT,
        capsize=3,
    )
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(-PRACTICAL_BAR, color=ACCENT_2, ls="--", lw=1, label="2 point bar")
    ax.set_yticks(y, week["tag_family"])
    ax.set_xlabel("Answer-rate difference, weekend minus weekday (within year)")
    ax.set_title("Weekend gap by tag family, train window (descriptive)")
    ax.legend(loc="lower right")
    finish(fig, FIGURES / "09_weekend_by_family.png")


def run_hypotheses(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df, _ = load_analysis()
    train = df.loc[df["in_primary"] & (df["split"] == "train")].copy()
    train["answered_24h"] = train["answered_24h"].astype(int)

    code = mantel_haenszel_rd(train, "has_code", strata=["tag_family", "year"])
    code_crude = mantel_haenszel_rd(train, "has_code", strata=None)
    weekend = mantel_haenszel_rd(train, "weekend", strata=["tag_family", "year"])
    weekend_crude = mantel_haenszel_rd(train, "weekend", strata=None)
    print("Fitting title-length logits on the train window...")
    title = _title_length_test(train)

    no_edit = train.loc[~train["has_edit_marker"]].copy()
    code_no_edit = mantel_haenszel_rd(no_edit, "has_code", strata=["tag_family", "year"])

    # Sensitivity: score fast closures as failures instead of dropping them.
    sens_rows = df.loc[df["split"] == "train"].copy()
    sens_rows = sens_rows.loc[sens_rows["in_primary"] | sens_rows["closed_fast_unanswered"]].copy()
    sens_rows["answered_24h"] = np.where(sens_rows["closed_fast_unanswered"], 0, sens_rows["answered_24h"].astype(int))
    code_with_closed = mantel_haenszel_rd(sens_rows, "has_code", strata=["tag_family", "year"])

    p_values = {
        "code_block": code["p_value"],
        "weekend": weekend["p_value"],
        "title_length": title["p_value"],
    }
    adjusted = holm(p_values)
    code_decision = _decision(code["rd"], code["ci95"], direction=1)
    weekend_decision = _decision(weekend["rd"], weekend["ci95"], direction=-1)
    title_practical = title["bin_rate_range"] >= PRACTICAL_BAR
    title_decision = (
        "non-linear and the bin gap clears 2 points"
        if title["p_value"] < 0.05 and title_practical
        else "non-linear but the bin gap is under 2 points"
        if title["p_value"] < 0.05
        else "a straight line is enough"
    )

    payload = {
        "window": "train",
        "n_train_primary": int(len(train)),
        "practical_bar": PRACTICAL_BAR,
        "code_block": {
            **code,
            "crude": code_crude,
            "decision": code_decision,
            "holm_p": adjusted["code_block"],
            "hypothesis": "Code blocks have a higher 24-hour answer rate within tag family and year.",
        },
        "weekend": {
            **weekend,
            "crude": weekend_crude,
            "decision": weekend_decision,
            "holm_p": adjusted["weekend"],
            "simpson_gap": float(weekend_crude["rd"] - weekend["rd"]),
            "hypothesis": "Weekend posts have a lower 24-hour answer rate within tag family and year.",
        },
        "title_length": {
            **title,
            "decision": title_decision,
            "holm_p": adjusted["title_length"],
            "hypothesis": "Title length is not linear: very short and very long titles both do worse.",
        },
        "robustness_code_without_edit_markers": {
            **code_no_edit,
            "n_train": int(len(no_edit)),
            "decision": _decision(code_no_edit["rd"], code_no_edit["ci95"], direction=1),
        },
        "sensitivity_code_closed_as_failure": {
            **code_with_closed,
            "n_train": int(len(sens_rows)),
            "decision": _decision(code_with_closed["rd"], code_with_closed["ci95"], direction=1),
        },
        "holm": adjusted,
    }
    write_json(OUTPUTS / "hypotheses.json", payload)
    _family_plot(train)
    _title_adjusted_plot(title)
    print(
        f"Code RD={code['rd']:+.4f} {code_decision}; "
        f"weekend RD={weekend['rd']:+.4f} {weekend_decision}; "
        f"title {title_decision}"
    )
    return payload


def _title_adjusted_plot(title: dict) -> None:
    apply_style()
    rates = title["year_standardized_bin_rates"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(list(rates.keys()), list(rates.values()), color=ACCENT, marker="o")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Year-standardized 24-hour answer rate")
    ax.set_xlabel("Title length (characters)")
    ax.set_title("Title-length shape after holding year fixed, train window")
    finish(fig, FIGURES / "10_title_length_adjusted.png")


if __name__ == "__main__":
    run_hypotheses()
