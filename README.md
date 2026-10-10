# Which developer questions go unanswered

A data-science project about unanswered questions in developer Q&A communities. Expert attention is limited, so the project predicts which new questions will still have no answer from someone else 24 hours later, and examines whether requiring a code sample could raise answer rates.

The data is the Kaggle set [StackSample: 10% of Stack Overflow Q&A](https://www.kaggle.com/datasets/stackoverflow/stacksample) (CC BY-SA). It is every question whose id is a multiple of 10, from August 2008 through October 2016, plus the answers and tags on those questions. That is 1,264,216 questions and 2,014,516 answers.

## Result in one paragraph

72.4% of questions received an answer from someone else within 24 hours. That rate fell from 92% in 2008 to 59% in 2016. A code block is associated with a 3.5 percentage-point higher answer rate after holding tag family and year fixed (95% CI 3.3 to 3.7). Weekend posting is not a problem. Longer titles are answered less often, and the drop flattens rather than turning back up. For a review queue that can touch 5% of new questions, a linear model on the title words puts truly unanswered questions in 69.5% of those slots on a later time window. Gradient boosting on structural features reaches 60.8%, a "no code and a rare tag" rule reaches 56.2%, and the base rate is 40.9%.

The code-block gap is an association, not the effect of requiring code. The experiment that would measure that effect is designed in the last section.

## The two decisions

Two product actions get confused in this kind of project, and they need different evidence.

1. **Rank.** Spend a scarce review slot on the questions most likely to die unanswered. The ranker may use anything known at post time, including the words in the title.
2. **Change the composer.** Require a code block, or an explicit "no code needed," before the question is posted. This needs a specific lever, and the lever has to be tested. "Titles containing `oauth` go unanswered" is a true ranking fact and a bad composer rule.

## The label

The label is mechanical: no answer from a different user within 24 hours of posting.

- Answers by the asker are ignored when both owner ids are present and equal.
- Questions closed within 24 hours with no other answer (7,546 rows, mostly duplicates that were redirected) are left out of the primary label and counted as failures only in a sensitivity check.
- Questions posted in the last 24 hours of the dump are dropped so the file's end date cannot create false failures.
- Score, view count, answer count, answer score, and closed date are outcomes. They are never features.

Among questions that are ever answered by someone else, the median wait is 21 minutes and 88% arrive inside 24 hours. The cutoff sits after the typical answer and before the long tail, which is what a "this one is stuck" label should do.

## What the data cannot support

These limits shape how every number below should be read.

- **Post text is the latest revision, not the original.** The dump stores the current title, body, and tags. People add code after a commenter asks for it, so "has code" can partly reflect having received help. The code-block gap is unchanged when posts with an explicit edit marker (`edit:` or `update:`, about 4% of training rows) are removed, but that check is incomplete, because many edits are not labeled.
- **No user timezone.** Hour of day is UTC and mixes countries. Permuting hour barely moves model performance.
- **No reputation at post time.** About 1.1% of owner ids are missing. A missing owner id is not a measure of being a new user.
- **A 10% sample of questions, not of users.** Ids are exactly the multiples of 10. A user who asked ten questions contributes about one row. Live-site volumes in the experiment design are scaled by 10 for this reason.
- **The dump ends in 2016.** Base rates will not match today's site. The decision logic transfers; the numbers do not.
- **Everything is observational.** The tests below are associations. The report does not call them causal.

## Three pre-specified tests

The tests were written down before looking at subgroups and estimated on the training window only (through 8 April 2015). Each risk difference is a Mantel–Haenszel weighted average within strata of tag family by year, with a normal 95% interval. With 879,000 rows a 0.6-point gap has a p-value near 10⁻⁷, so the p-value is not the decision. The decision is whether the interval clears a 2-percentage-point practical bar. Holm adjustment across the three tests changes no call.

| Test | Adjusted difference | 95% CI | Call |
| --- | ---: | ---: | --- |
| Code block vs. none | +3.51 pts | +3.32 to +3.71 | Clears the 2-point bar |
| Weekend vs. weekday | +0.60 pts | +0.37 to +0.83 | Opposite direction, under the bar |
| Title length, quadratic vs. linear | LR = 201.5 on 1 df | | Non-linear; the decline flattens |

Robustness for the code-block gap: +3.65 without edit-marked posts, +3.59 with fast closures counted as failures. The crude gap is +3.64, so tag and year mix is not creating the result.

The weekend result is the example of why the interval matters. Holding tag mix fixed moves it by a quarter of a point in the same direction, so it is not a Simpson artifact. It is a small positive association and nowhere near actionable.

Year-standardized 24-hour answer rate by title length:

| Title length | Answer rate |
| --- | ---: |
| 1–20 characters | 82.8% |
| 21–40 | 79.6% |
| 41–60 | 76.6% |
| 61–80 | 73.7% |
| 81–100 | 71.6% |
| 101+ | 70.3% |

The pre-specified story was a U-shape. The short titles do best, and the curve falls then flattens. That is why a squared term improves the fit and why a single slope is slightly wrong.

## Models

The positive class is "still unanswered at 24 hours." The time split is 70% / 15% / 15% by posting date. Models were chosen on the validation window and scored once on the test window (after 28 January 2016). The operating point is the top 5% of predicted risk, because the queue has a capacity, not a 0.5 threshold.

| Model | Test precision at 5% | Test ROC-AUC | Test Brier |
| --- | ---: | ---: | ---: |
| Base rate | 40.9% | 0.500 | 0.273 |
| Rule: no code and a tag outside the top 15 | 56.2% | 0.546 | — |
| L2 logistic regression | 59.0% | 0.637 | 0.251 |
| Random forest | 60.2% | 0.639 | 0.245 |
| Histogram gradient boosting | 60.8% | 0.641 | 0.250 |
| TF-IDF logistic on title words | 69.5% | 0.699 | 0.218 |

Why each one is on the list:

- **Base rate** sets the floor and shows drift: its Brier score worsens from 0.235 on validation to 0.273 on test because the unanswered share moved from 33.9% to 40.9%.
- **The rule** is what a model has to beat before anyone should maintain it.
- **L2 logistic regression** is the explainable model. Each coefficient is the change in log-odds with the other features held fixed. L2 keeps rare tag levels from getting absurd weights.
- **Random forest** averages deep trees and cuts variance. It is the contrast arm against boosting, which attacks bias instead.
- **Gradient boosting** is allowed to use the title-length bend and feature interactions that logistic regression cannot. It wins among structural models by a small margin, which means the bend and the interactions are small.
- **Title words** ask whether the words beat the structure. They do, by about nine points of queue precision. Words weighted toward going unanswered include `plugin`, `oauth`, `api`, `grunt`, `bluetooth`. Words weighted the other way include `regex`, `array`, `string`, `div`, `css`. Integration questions have a thin set of answerers; language-basics questions have a thick one.
- **Left off on purpose.** k-nearest neighbors has no sensible distance across tags. A support-vector machine does not return a probability the queue can threshold. A neural net is a poor default here because the winning signal is sparse title text, which a linear model already handles with readable weights.

Permutation importance for the boosting model on the test window: tag family (AUC drop about 0.08), code-block count (0.05), body length (0.04). Weekend, hour, and the binary has-code flag add nothing once the count is in the model. Year is unimportant inside the test window because that window is one year; its effect is the drift between windows.

Calibration was fit on the validation window and does not hold on the test window: a predicted probability near 0.45 shows up as an observed rate near 0.60, because the base rate moved. The scores rank the queue well, and the probabilities need recalibrating as the base rate drifts.

## What this supports shipping

- Rank the 5% review queue with the title-word score. On the test window that queue holds about 9,400 questions, of which about 6,500 really go unanswered, against about 3,900 if the slots were random.
- Keep logistic regression over gradient boosting for the structural features. The lift is about two points and logistic regression can be read aloud.
- Do not change anything about weekends.
- Take the code-block association to an experiment, not to a forced wizard.

## The experiment this data cannot run

The design in `src/experiment.py`:

- **Scope:** JavaScript, Java, C#, PHP, and Android, where the recent 24-hour answer rate is 62.2%.
- **Unit of randomization:** users, not questions. One person posts many questions, so question-level randomization is not an independent sample. The intraclass correlation within users is about 0.10.
- **Treatment:** the composer requires a code block or an explicit "no code needed" before posting.
- **Primary metric:** 24-hour answer rate from someone else. Minimum effect worth detecting: 2 percentage points, two-sided 5% test, 80% power.
- **Sample size:** about 9,100 questions per arm ignoring clustering; about 17,800 per arm after a design effect of 1.95 from users posting roughly 10 questions each on the full site. At about 2,800 questions a day across those tags, that arrives in about 13 days. Run 14 days anyway so the first week's novelty is not the whole result.
- **Guardrail:** questions successfully posted per randomized user. Do not ship if the treatment arm posts more than 5% fewer questions and the interval excludes zero. A harsher composer can "win" the answer rate by scaring off the questions that would have failed.

## How to run

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
python -m src.download          # or copy Questions.csv, Answers.csv, Tags.csv into data/raw/
python -m src.run_all --force-build
```

`python -m src.run_all` skips stages whose outputs already exist. Add `--force` to recompute them. The narrative notebook is [notebooks/analysis.ipynb](notebooks/analysis.ipynb). It calls the same `src` functions.

Figures and numeric results land in `outputs/`. The raw CSVs are large and are not committed.

## What is pre-specified

These choices are fixed in [src/config.py](src/config.py) and are not tuned on the test window:

- The label is no answer from someone other than the asker within 24 hours.
- A 2 percentage-point gap is the practical bar for changing the composer.
- The three tests are the code-block gap, the weekend gap, and a linear-versus-quadratic title-length comparison, with Holm adjustment across them.
- The time split is 70% / 15% / 15% by posting time. Tag families are the 15 most common tags in the training window; everything else is `other`.
- The queue metric is precision in the top 5% of predicted risk. The model is chosen on the validation window and scored once on the test window.

## Layout

| Path | Role |
| --- | --- |
| `src/build.py` | Read the CSVs and build the label in SQL |
| `src/eda.py` | Exploration figures |
| `src/hypotheses.py` | Stratified risk differences and the title-length test |
| `src/models.py` | Baselines, logistic regression, random forest, boosting, title words |
| `src/experiment.py` | Sample size for a user-randomized composer test |
| `notebooks/analysis.ipynb` | Narrative notebook calling the `src` functions |
| `outputs/` | Figures and JSON results from the last run |
