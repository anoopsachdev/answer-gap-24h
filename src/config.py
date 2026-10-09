"""Paths and pre-specified analysis choices."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TMP = ROOT / "data" / "tmp"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"

QUESTIONS = RAW / "Questions.csv"
ANSWERS = RAW / "Answers.csv"
TAGS = RAW / "Tags.csv"
PARQUET = PROCESSED / "questions.parquet"

KAGGLE_DATASET = "stackoverflow/stacksample"
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/stackoverflow/stacksample"

# Pre-specified before looking at outcomes. Do not tune these on the test window.
ANSWER_WINDOW_HOURS = 24
PRACTICAL_BAR = 0.02
HOLM_FAMILY = ("code_block", "weekend", "title_length")
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.85  # train | val | test cutpoints on the primary-set time distribution
TOP_TAGS = 15
MIN_STRATUM_N = 20
QUEUE_FRACTION = 0.05
EXPERIMENT_TAGS = 5
EXPERIMENT_ALPHA = 0.05
EXPERIMENT_POWER = 0.80
EXPERIMENT_MDE = 0.02
GUARDRAIL_RELATIVE_DROP = 0.05
NOVELTY_DAYS = 14
SAMPLE_FRACTION = 0.10  # question ids that are multiples of 10
RANDOM_STATE = 0
EDIT_PATTERN = r"(edit|update)\s*:|\*\*\s*edit"
