# Which developer questions go unanswered

A data-science project about unanswered questions in developer Q&A communities. Expert time is limited, so the project predicts which new questions will still have no answer from someone else 24 hours later and examines whether requiring a code sample could improve answer rates.

The data is the Kaggle set [StackSample: 10% of Stack Overflow Q&A](https://www.kaggle.com/datasets/stackoverflow/stacksample) (CC BY-SA). It is every question whose id is a multiple of 10, from August 2008 through October 2016, plus the answers and tags on those questions.

## Result in one paragraph

On 1.26 million questions, 72.4% received an answer from someone else within 24 hours. That rate fell from 92% in 2008 to 59% in 2016. A code block is associated with a 3.5 percentage-point higher answer rate after holding tag family and year fixed (95% CI 3.3 to 3.7). Weekend posting is not a problem. Longer titles are answered less often, and the drop flattens rather than turning back up. For a queue that can touch 5% of new questions, a linear model on the title words puts truly unanswered questions in 69.5% of those slots on a later time window. A structural gradient-boosting model reaches 60.8%. A "no code and a rare tag" rule reaches 56.2%. The base rate is 40.9%.

## How to run

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
python -m src.download          # or copy Questions.csv, Answers.csv, Tags.csv into data/raw/
python -m src.run_all --force-build
```

`python -m src.run_all` skips stages whose outputs already exist. Add `--force` to recompute them. The narrative notebook is [notebooks/analysis.ipynb](notebooks/analysis.ipynb). It calls the same `src` functions.

Figures and the numeric results land in `outputs/`. The raw CSVs are large and are not committed.

## What is pre-specified

These choices are fixed in [src/config.py](src/config.py) and are not tuned on the test window:

- The label is no answer from someone other than the asker within 24 hours.
- A 2 percentage-point gap is the practical bar for changing the wizard.
- The three tests are the code-block gap, the weekend gap, and a linear-versus-quadratic title-length comparison. Holm adjustment is applied across those three p-values. The decision uses the interval and the bar, because a million rows makes tiny gaps "significant."
- The time split is 70% / 15% / 15% by posting time. Tag families are the 15 most common tags in the training window, and everything else is `other`.
- The queue metric is precision in the top 5% of predicted risk. The model is chosen on the validation window and scored once on the test window.

## Layout

| Path | Role |
| --- | --- |
| `src/build.py` | Read the CSVs and build the label in SQL |
| `src/eda.py` | Exploration figures |
| `src/hypotheses.py` | Stratified risk differences and the title-length test |
| `src/models.py` | Baselines, logistic regression, random forest, boosting, title words |
| `src/experiment.py` | Sample size for a user-randomized wizard test |
