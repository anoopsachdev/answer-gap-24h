"""Load the analysis table and apply the pre-specified time split."""

from __future__ import annotations

import pandas as pd

from src.config import OUTPUTS, PARQUET, TOP_TAGS, TRAIN_FRACTION, VAL_FRACTION
from src.io_utils import read_json, write_json


def load_frame() -> pd.DataFrame:
    if not PARQUET.exists():
        raise FileNotFoundError("Run the label build before loading the analysis table.")
    df = pd.read_parquet(PARQUET)
    df["created"] = pd.to_datetime(df["created"], utc=True)
    df["has_code"] = df["code_block_count"].fillna(0).astype(int) > 0
    df["weekend"] = df["weekend"].astype(bool)
    df["answered_24h"] = df["answered_24h"].astype(bool)
    df["unanswered_24h"] = df["unanswered_24h"].astype(bool)
    df["in_primary"] = df["in_primary"].astype(bool)
    df["has_edit_marker"] = df["has_edit_marker"].astype(bool)
    df["closed_fast_unanswered"] = df["closed_fast_unanswered"].astype(bool)
    df["eventually_answered"] = df["eventually_answered"].astype(bool)
    return df


def assign_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cut the primary set by time. Tag families are learned on the train window only."""
    primary = df.loc[df["in_primary"], "created"]
    t_train = primary.quantile(TRAIN_FRACTION)
    t_val = primary.quantile(VAL_FRACTION)
    out = df.copy()
    out["split"] = "test"
    out.loc[out["created"] <= t_val, "split"] = "val"
    out.loc[out["created"] <= t_train, "split"] = "train"

    train_tags = (
        out.loc[out["in_primary"] & (out["split"] == "train"), "primary_tag"]
        .value_counts()
        .head(TOP_TAGS)
        .index.tolist()
    )
    known = set(train_tags)
    out["tag_family"] = out["primary_tag"].where(out["primary_tag"].isin(known), "other")

    meta = {
        "train_end": t_train,
        "val_end": t_val,
        "tag_families": train_tags + ["other"],
        "n_by_split_primary": (
            out.loc[out["in_primary"]].groupby("split").size().astype(int).to_dict()
        ),
    }
    write_json(OUTPUTS / "split.json", meta)
    return out, meta


def load_analysis() -> tuple[pd.DataFrame, dict]:
    df, meta = assign_splits(load_frame())
    return df, meta


def split_meta() -> dict:
    return read_json(OUTPUTS / "split.json")
