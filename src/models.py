"""Rank questions that will still be unanswered at 24 hours.

Model choice is made on the validation window. The test window is scored once.
The positive class is the failure: unanswered at 24 hours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import FIGURES, MODELS, OUTPUTS, QUEUE_FRACTION, RANDOM_STATE
from src.data import load_analysis
from src.io_utils import write_json
from src.plots import ACCENT, ACCENT_2, ACCENT_3, MUTED, apply_style, finish

import matplotlib.pyplot as plt

CONT_COLS = [
    "title_len",
    "body_len",
    "code_block_count",
    "link_count",
    "tag_count",
    "hour_sin",
    "hour_cos",
]
BIN_COLS = ["has_code", "weekend"]
CAT_COLS = ["tag_family", "year"]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[df["in_primary"]].copy()
    out["has_code"] = out["has_code"].astype(float)
    out["weekend"] = out["weekend"].astype(float)
    out["year"] = out["year"].astype(str)
    radians = 2 * np.pi * out["hour_utc"].astype(float) / 24.0
    out["hour_sin"] = np.sin(radians)
    out["hour_cos"] = np.cos(radians)
    out["y"] = out["unanswered_24h"].astype(int)
    return out


def _cap(train: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    caps = {
        "body_len": float(train["body_len"].quantile(0.99)),
        "title_len": float(train["title_len"].quantile(0.995)),
        "link_count": float(train["link_count"].quantile(0.99)),
        "code_block_count": 10.0,
    }
    out = frame.copy()
    for col, cap in caps.items():
        out[col] = out[col].clip(upper=cap).astype(float)
    return out, caps


def precision_at_fraction(y_true: np.ndarray, scores: np.ndarray, fraction: float = QUEUE_FRACTION) -> float:
    k = max(1, int(round(fraction * len(y_true))))
    jitter = np.random.default_rng(RANDOM_STATE).random(len(scores)) * 1e-9
    order = np.argsort(-(scores + jitter))
    return float(y_true[order[:k]].mean())


def _metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    scores = np.clip(scores, 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(y_true, scores)),
        "log_loss": float(log_loss(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision_at_5pct": precision_at_fraction(y_true, scores),
        "prevalence": float(np.mean(y_true)),
    }


def _maybe_calibrate(y_val: np.ndarray, raw_val: np.ndarray, raw_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_val, y_val)
    cal_val = iso.predict(raw_val)
    if brier_score_loss(y_val, cal_val) + 0.001 < brier_score_loss(y_val, raw_val):
        return cal_val, iso.predict(raw_test), True
    return raw_val, raw_test, False


def _structure_matrix():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), CONT_COLS),
            ("bin", "passthrough", BIN_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_COLS,
            ),
        ]
    )


def _fit_hgb(X_train, y_train, X_val, y_val) -> HistGradientBoostingClassifier:
    """Grow trees, then keep the iteration count that minimized val log loss."""
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    best_iter = 1
    best_loss = np.inf
    wait = 0
    for i, proba in enumerate(model.staged_predict_proba(X_val), start=1):
        loss = log_loss(y_val, proba[:, 1])
        if loss < best_loss - 1e-4:
            best_loss = loss
            best_iter = i
            wait = 0
        else:
            wait += 1
            if wait >= 15:
                break
    if best_iter < 200:
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=best_iter,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
    model.best_iteration_ = best_iter
    return model


def _rule_scores(frame: pd.DataFrame) -> np.ndarray:
    return ((frame["has_code"] == 0) & (frame["tag_family"] == "other")).astype(float).to_numpy()


def run_models(df: pd.DataFrame | None = None) -> dict:
    apply_style()
    if df is None:
        df, _ = load_analysis()
    prepared = _prepare(df)
    parts = {name: prepared.loc[prepared["split"] == name].copy() for name in ("train", "val", "test")}
    capped = {}
    caps = None
    for name, part in parts.items():
        capped[name], caps = _cap(parts["train"], part)
    train, val, test = capped["train"], capped["val"], capped["test"]

    pre = _structure_matrix()
    X_train = pre.fit_transform(train)
    X_val = pre.transform(val)
    X_test = pre.transform(test)
    y_train = train["y"].to_numpy()
    y_val = val["y"].to_numpy()
    y_test = test["y"].to_numpy()

    print(f"Train {len(train):,}  val {len(val):,}  test {len(test):,}")
    scores_val: dict[str, np.ndarray] = {}
    scores_test: dict[str, np.ndarray] = {}
    fitted = {}

    base = float(y_train.mean())
    scores_val["base_rate"] = np.full(len(y_val), base)
    scores_test["base_rate"] = np.full(len(y_test), base)

    scores_val["business_rule"] = _rule_scores(val)
    scores_test["business_rule"] = _rule_scores(test)

    print("Fitting L2 logistic regression...")
    logit = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=RANDOM_STATE)
    logit.fit(X_train, y_train)
    fitted["logistic"] = logit
    scores_val["logistic"] = logit.predict_proba(X_val)[:, 1]
    scores_test["logistic"] = logit.predict_proba(X_test)[:, 1]

    print("Fitting random forest...")
    forest = RandomForestClassifier(
        n_estimators=80,
        min_samples_leaf=40,
        max_depth=24,
        max_features="sqrt",
        max_samples=150_000,
        n_jobs=2,
        random_state=RANDOM_STATE,
    )
    forest.fit(X_train, y_train)
    fitted["random_forest"] = forest
    scores_val["random_forest"] = forest.predict_proba(X_val)[:, 1]
    scores_test["random_forest"] = forest.predict_proba(X_test)[:, 1]

    print("Fitting histogram gradient boosting...")
    boosting = _fit_hgb(X_train, y_train, X_val, y_val)
    fitted["gradient_boosting"] = boosting
    scores_val["gradient_boosting"] = boosting.predict_proba(X_val)[:, 1]
    scores_test["gradient_boosting"] = boosting.predict_proba(X_test)[:, 1]
    print(f"Boosting stopped at {boosting.best_iteration_} iterations")

    print("Fitting title TF-IDF logistic...")
    tfidf = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=10),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-5,
                    random_state=RANDOM_STATE,
                    max_iter=30,
                ),
            ),
        ]
    )
    tfidf.fit(train["title"].fillna(""), y_train)
    fitted["title_tfidf"] = tfidf
    scores_val["title_tfidf"] = tfidf.predict_proba(val["title"].fillna(""))[:, 1]
    scores_test["title_tfidf"] = tfidf.predict_proba(test["title"].fillna(""))[:, 1]

    calibrated = {}
    for name in ("logistic", "random_forest", "gradient_boosting", "title_tfidf"):
        val_s, test_s, used = _maybe_calibrate(y_val, scores_val[name], scores_test[name])
        scores_val[name] = val_s
        scores_test[name] = test_s
        calibrated[name] = used

    metrics = {}
    for name in scores_val:
        metrics[name] = {
            "val": _metrics(y_val, scores_val[name]),
            "test": _metrics(y_test, scores_test[name]),
            "calibrated": calibrated.get(name, False),
        }

    candidates = ["business_rule", "logistic", "random_forest", "gradient_boosting", "title_tfidf"]
    winner = max(candidates, key=lambda name: metrics[name]["val"]["precision_at_5pct"])
    rule_val = metrics["business_rule"]["val"]["precision_at_5pct"]
    winner_val = metrics[winner]["val"]["precision_at_5pct"]
    ships = winner != "business_rule" and winner_val >= rule_val + 0.01

    structural = ["logistic", "random_forest", "gradient_boosting"]
    best_structural = max(structural, key=lambda name: metrics[name]["val"]["precision_at_5pct"])

    print("Permutation importance and partial dependence...")
    structural_pipe = Pipeline([("pre", pre), ("model", fitted[best_structural])])
    importance = _importance(structural_pipe, test)
    _plots(metrics, scores_test, y_test, structural_pipe, test, importance, best_structural)
    _word_plot(fitted["title_tfidf"])

    no_edit = _edit_robustness(train, val, test, pre)

    payload = {
        "caps": caps,
        "positive_class": "unanswered_24h",
        "queue_fraction": QUEUE_FRACTION,
        "metrics": metrics,
        "selected_on": "validation precision at 5%",
        "selected_model": winner,
        "ships": ships,
        "ship_reason": (
            f"{winner} beats the business rule by at least 1 point of precision at 5% on the validation window."
            if ships
            else "No model beat the business rule by 1 point of validation precision at 5%, so the ranker would not ship."
        ),
        "best_structural_model": best_structural,
        "boosting_iterations": int(boosting.best_iteration_),
        "forest_max_samples": 150000,
        "edit_marker_robustness": no_edit,
        "feature_notes": {
            "numeric": CONT_COLS + BIN_COLS,
            "categorical": CAT_COLS,
            "banned": ["score", "answer_count", "answer_score", "closed_date"],
        },
    }
    write_json(OUTPUTS / "model_metrics.json", payload)
    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"Selected {winner}; ships={ships}")
    for name, row in metrics.items():
        test_m = row["test"]
        print(
            f"  {name:20} val P@5={row['val']['precision_at_5pct']:.3f}  "
            f"test P@5={test_m['precision_at_5pct']:.3f}  "
            f"AUC={test_m['roc_auc']:.3f}  Brier={test_m['brier']:.3f}"
        )
    return payload


def _importance(model, test: pd.DataFrame) -> dict:
    cols = CONT_COLS + BIN_COLS + CAT_COLS
    sample = test.sample(n=min(12000, len(test)), random_state=RANDOM_STATE)
    result = permutation_importance(
        model,
        sample[cols],
        sample["y"].to_numpy(),
        n_repeats=4,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=2,
    )
    order = np.argsort(result.importances_mean)[::-1][:12]
    return {
        "features": [cols[i] for i in order],
        "mean_auc_drop": [float(result.importances_mean[i]) for i in order],
    }


def _plots(metrics, scores_test, y_test, model, test, importance, structural_name) -> None:
    names = ["base_rate", "business_rule", "logistic", "random_forest", "gradient_boosting", "title_tfidf"]
    labels = ["Base rate", "Rule", "Logistic", "Forest", "Boosting", "Title words"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(names))
    vals = [metrics[name]["test"]["precision_at_5pct"] for name in names]
    colors = [MUTED, ACCENT_2, ACCENT, ACCENT_3, ACCENT, ACCENT_3]
    ax.bar(x, vals, color=colors)
    ax.axhline(metrics["base_rate"]["test"]["prevalence"], color=MUTED, ls="--", lw=1, label="Prevalence")
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("Precision in the top 5% of the test window")
    ax.set_ylim(0, 1)
    ax.set_title("Queue precision on the test window (model was chosen on validation)")
    ax.legend()
    finish(fig, FIGURES / "11_precision_at_capacity.png")

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    for name, color in (
        ("logistic", ACCENT),
        ("random_forest", ACCENT_3),
        ("gradient_boosting", ACCENT_2),
    ):
        frac, mean = calibration_curve(y_test, scores_test[name], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", color=color, label=name.replace("_", " "))
    ax.set_xlabel("Predicted probability of still being unanswered")
    ax.set_ylabel("Observed fraction unanswered")
    ax.set_title("Reliability on the test window")
    ax.legend()
    finish(fig, FIGURES / "12_calibration.png")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    order = np.argsort(importance["mean_auc_drop"])
    ax.barh(
        [importance["features"][i] for i in order],
        [importance["mean_auc_drop"][i] for i in order],
        color=ACCENT,
    )
    ax.set_xlabel("Drop in ROC-AUC when the column is shuffled")
    ax.set_title(f"Permutation importance, {structural_name.replace('_', ' ')}, test sample")
    finish(fig, FIGURES / "13_importance.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    sample = test.sample(n=min(8000, len(test)), random_state=RANDOM_STATE)
    PartialDependenceDisplay.from_estimator(
        model,
        sample[CONT_COLS + BIN_COLS + CAT_COLS],
        features=["title_len", "code_block_count"],
        kind="average",
        ax=axes,
    )
    fig.suptitle(f"Partial dependence, {structural_name.replace('_', ' ')}")
    finish(fig, FIGURES / "14_partial_dependence.png")


def _word_plot(pipeline: Pipeline) -> None:
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf: SGDClassifier = pipeline.named_steps["clf"]
    names = np.array(vectorizer.get_feature_names_out())
    coef = clf.coef_.ravel()
    top = np.argsort(coef)[-8:]
    bottom = np.argsort(coef)[:8]
    chosen = np.concatenate([bottom, top])
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colors = [ACCENT_2 if coef[i] < 0 else ACCENT for i in chosen]
    ax.barh(names[chosen], coef[chosen], color=colors)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("L2-logistic weight on the title (positive = more often unanswered)")
    ax.set_title("Title words associated with going unanswered")
    finish(fig, FIGURES / "15_title_words.png")


def _edit_robustness(train, val, test, pre) -> dict:
    """Refit logistic and boosting without posts that contain an edit marker."""
    mask = {
        "train": ~train["has_edit_marker"].to_numpy(),
        "val": ~val["has_edit_marker"].to_numpy(),
        "test": ~test["has_edit_marker"].to_numpy(),
    }
    tr, va, te = train.loc[mask["train"]], val.loc[mask["val"]], test.loc[mask["test"]]
    encoder = _structure_matrix()
    Xtr = encoder.fit_transform(tr)
    Xva = encoder.transform(va)
    Xte = encoder.transform(te)
    ytr, yva, yte = tr["y"].to_numpy(), va["y"].to_numpy(), te["y"].to_numpy()
    logit = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=RANDOM_STATE)
    logit.fit(Xtr, ytr)
    boosting = _fit_hgb(Xtr, ytr, Xva, yva)
    out = {}
    for name, model in (("logistic", logit), ("gradient_boosting", boosting)):
        raw_val = model.predict_proba(Xva)[:, 1]
        raw_test = model.predict_proba(Xte)[:, 1]
        val_s, test_s, used = _maybe_calibrate(yva, raw_val, raw_test)
        out[name] = {
            "val": _metrics(yva, val_s),
            "test": _metrics(yte, test_s),
            "calibrated": used,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
        }
    return out


if __name__ == "__main__":
    run_models()
