"""Download the StackSample CSVs from Kaggle's public dataset endpoint.

The endpoint redirects to a short-lived storage URL and did not require a
Kaggle login when this project was built. If it starts asking for one, place
Questions.csv, Answers.csv, and Tags.csv in data/raw/ yourself.
"""

import subprocess
import zipfile

from src.config import KAGGLE_URL, QUESTIONS, RAW


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if QUESTIONS.exists() and (RAW / "Answers.csv").exists() and (RAW / "Tags.csv").exists():
        print("CSVs already present; skipping download.")
        return
    archive = RAW / "archive.zip"
    print(f"Downloading {KAGGLE_URL}")
    subprocess.run(["curl", "-L", "--fail", "-o", str(archive), KAGGLE_URL], check=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(RAW)
    archive.unlink(missing_ok=True)
    print("Extracted CSVs into", RAW)


if __name__ == "__main__":
    download()
