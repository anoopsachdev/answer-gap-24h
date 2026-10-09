# Interview brief: which questions go unanswered

## Two minutes

A developer Q&A site only works if a new question gets an answer from someone else while the asker still cares. This file is a 10% sample of Stack Overflow, 1,264,216 questions from August 2008 to October 2016. The label is mechanical: no answer from a different user within 24 hours. Score, view count, and answer count are outcomes, so they are not features.

72.4% of questions in the analysis set are answered inside that window. Among questions that are ever answered by someone else, the median wait is 21 minutes, and 88% of those answers arrive inside 24 hours. The cutoff sits after the typical answer and before the long tail.

The 24-hour answer rate falls from 92% in 2008 to 59% in 2016. Any model that ignores time will look better than it is, and any claim about "the" answer rate has to say which year.

Three tests were written down before looking at subgroups, and estimated on the training window only (through 8 April 2015), with tag family and year held fixed:

- A question with a code block is answered 3.5 percentage points more often (95% CI 3.3 to 3.7). That clears a 2-point bar for bothering to change the composer. Dropping posts whose body contains an edit marker leaves the gap at 3.7 points. Counting fast closures as failures leaves it at 3.6.
- Weekend posts are answered 0.6 points more often (CI 0.4 to 0.8), not less. The gap is far under the 2-point bar. Holding the tag mix fixed shrinks it only slightly, so this is not a Simpson artifact. The weekend hypothesis does not survive.
- Longer titles are answered less often. Year-standardized rates fall from 82.8% at 1–20 characters to 70.3% above 100 characters. A quadratic term fits better than a straight line. The shape is a decline that flattens, not a U.

The product use of a model is a queue: staff or an automated nudge can touch about 5% of new questions. On the test window (after 28 January 2016), 40.9% of questions go unanswered. A "no code, and a tag outside the top 15" rule puts unanswered questions in 56.2% of the queue. L2 logistic regression reaches 59.0%, a random forest 60.2%, gradient boosting 60.8%, and a linear model on the title words 69.5%. The title-word model was chosen on the validation window, then scored once.

The code-block gap is an association. The experiment that would test a wizard is sized below: about 18,000 questions per arm, randomized by user, run for 14 days in JavaScript, Java, C#, PHP, and Android.

## The decision

Two different actions get confused in this kind of project.

1. **Rank.** Spend a scarce review slot on the questions most likely to die unanswered.
2. **Change the composer.** Require a code block, or an explicit "no code needed," before the question is posted.

The ranker is allowed to use whatever is known at post time, including the words in the title. The wizard decision needs a specific lever. "Titles that say `oauth` go unanswered" is a true ranking fact and a bad composer rule, because we cannot tell people to stop having OAuth problems. "Questions with code are answered more often" is a lever, and it is still only an association until an experiment.

## What the data can and cannot support

The dump stores the latest revision of the title, body, and tags, not the text at the moment of posting. People add code after someone asks for it. That can make "has code" look helpful because the code arrived with the help. Two checks limit the damage. The code-block gap is unchanged when posts with an explicit edit marker (`edit:` or `update:`) are removed, about 4% of the training rows. And the marker check is incomplete: many edits are not labeled. Say that if asked.

There is no user timezone. Hour is UTC, so it mixes countries. Permuting hour barely moves the test AUC.

There is no reputation-at-post-time. About 1.1% of owner ids are missing. A missing owner is not a measure of being a new user.

Question ids are exactly the multiples of 10, so this is a 1-in-10 sample of questions, not a sample of users. A user who asked ten questions contributes about one row. Live-site volume in the experiment section is scaled by 10 for that reason.

Self-answers are ignored when both owner ids are present and equal. A question closed within 24 hours with no other answer is left out of the main label (most of those are duplicates, which were redirected) and put back, as failures, only in a sensitivity check. Questions posted in the last 24 hours of the dump are dropped so the file's end date cannot create false failures. That removes under a thousand rows.

Score is banned. So is the number of answers, the score of those answers, and the closed date as a predictor.

## The tests

The outcome in the tests is the 24-hour answer rate. The model section flips it, because the queue ranks failures. The risk difference is a Mantel–Haenszel weighted average of the within-stratum differences, with strata of tag family by year. Strata with fewer than 20 rows in either arm are dropped. The interval is the normal interval from those weighted two-proportion variances.

With 879,000 training rows, a 0.6-point gap has a p-value around 10⁻⁷. That p-value is not a reason to ship a change. The rule used here: the 95% interval has to lie entirely past a 2-point bar. Holm adjustment across the three tests does not change any call, and it is reported because the family was pre-specified, not because it is the decision.

**Code.** With a code block, the training answer rate is 77.9%. Without one, 74.3%. The adjusted difference is +3.51 points (CI +3.32 to +3.71), on 125 strata covering essentially the whole training window. The crude gap is +3.64, so the tag and year mix is not creating the result. The same estimate is +3.65 without edit-marked posts and +3.59 if fast closures count as failures.

**Weekend.** Adjusted difference, weekend minus weekday: +0.60 points (CI +0.37 to +0.83). Crude is +0.86. Stratifying moves the estimate by a quarter of a point in the same direction. Weekend posting is a small positive association and well under the bar. Do not build a weekend intervention from this file.

**Title length.** A logistic model of the answer indicator on title length and year, versus the same model plus the square of title length. The likelihood-ratio statistic is 201.5 on 1 degree of freedom. The year-standardized rates by bin are:

| Title length | Answer rate |
| --- | ---: |
| 1–20 characters | 82.8% |
| 21–40 | 79.6% |
| 41–60 | 76.6% |
| 61–80 | 73.7% |
| 81–100 | 71.6% |
| 101+ | 70.3% |

The pre-specified story was that very short and very long titles would both do worse. The short titles do best. The curve falls and then flattens, which is why the square term matters, and why a single slope is a bit wrong. It is not a U.

## Why these algorithms

The positive class for the models is "still unanswered at 24 hours." Scores are probabilities. The operating point is the top 5% of the score, not 0.5, because the queue has a capacity. All of the numbers below are on the test window unless noted. Calibration, when it helped validation Brier by at least 0.001, is an isotonic map fit on the validation window only.

**Base rate.** Predict the training prevalence for every row. ROC-AUC is 0.50 by construction. Precision at 5% equals the test prevalence, 40.9%, because the ranking is random. Brier score rises from 0.235 on validation to 0.273 on test: the training prevalence is stale. That is the monitoring point. The unanswered share itself moves from 33.9% in the validation window to 40.9% in the test window.

**Business rule.** Flag "no code block, and a tag outside the 15 training-window families." Precision at 5% is 56.2%. Its log loss is meaningless, because the score is a hard 0/1 rather than a probability. The rule is the thing a model has to beat before anyone should maintain it. The bar used for shipping was one extra point of validation precision. Every structural model clears that. The title-word model clears it by a wide margin.

**L2 logistic regression.** The log-odds of going unanswered are a weighted sum of scaled lengths, the code-block count, tag count, sine and cosine of the UTC hour, a weekend flag, and one-hot tag family and year. L2 pulls the weights of rare tag levels toward zero; without it, a tag that appears a handful of times gets an absurd coefficient. A coefficient is the change in log-odds when that column moves and the others in the model stay fixed. It is the model you can read to a product manager. It is the wrong last model if the title-length curve bends, or if "Python, no code, Sunday" is worse than the sum of those three pieces. Test precision at 5% is 59.0%, ROC-AUC 0.637, Brier 0.251.

**Random forest.** Eighty deep trees, each on a 150,000-row bootstrap sample, with at least 40 rows in a leaf. Averaging trees cuts variance. It does not, by itself, go after the bias of a shallow tree the way boosting does. That is the comparison the forest is here to make. Test precision at 5% is 60.2%, AUC 0.639, Brier 0.245. It is close to boosting and a little better on Brier. Vote fractions from a forest are not probabilities until they are calibrated; the reported scores are the isotonic ones.

**Histogram gradient boosting.** Shallow trees (at most 31 leaves) are fit to the remaining log-loss, then added with a learning rate of 0.08. A small learning rate, a floor of 50 rows per leaf, and L2 on the leaves keep one odd branch from dominating. The iteration count, 108, is the one that minimized log loss on the validation window, with a patience of 15 rounds. This is the model that is allowed to use the bend in title length and interactions the logistic model misses. Test precision at 5% is 60.8%, AUC 0.641, Brier 0.250. It wins among the structural models, by a small margin. Refitting it without edit-marked posts leaves test precision at 60.7%. The leakage check does not move it.

On a test sample, shuffling tag family drops the boosting model's ROC-AUC by about 0.08, code-block count by about 0.05, and body length by about 0.04. The binary "has code" flag adds nothing once the count is in the model. Weekend and hour add nothing. Year adds nothing inside the test window because that window is almost all of 2016; the year effect is the drift between windows, which the base-rate Brier already shows. Partial dependence for the same model: as the title grows from about 30 to 90 characters, predicted risk rises from about 0.19 to about 0.24. Moving from no code block to four or more cuts predicted risk from about 0.27 to about 0.16. That contrast is wider than the binary test, which only compares any code with none.

**Title words.** A TF-IDF logistic regression on the title only, unigrams and bigrams, 15,000 features, L2, fit with SGD. This is on the list as a comparison, to see whether the words beat the structure. They do. Validation precision at 5% is 64.3%, test is 69.5%, AUC 0.699, Brier 0.218. Words whose weight points toward going unanswered include `plugin`, `oauth`, `api`, `grunt`, `bluetooth`. Words that point the other way include `regex`, `array`, `string`, `div`, `css`. The first group is integration work with a thin set of answerers. The second group is ordinary language and library questions with a thick set. That is a ranking feature. It is not, by itself, a composer rule.

The reliability curve on the test window does not sit on the diagonal. Isotonic calibration was fit while 33.9% of questions were going unanswered. In the test window that share is 40.9%. A predicted probability around 0.45 shows up as an observed rate around 0.60. Use the score to rank the queue. Recalibrate the probability when the base rate moves, which is the same fact the base-rate Brier score is already telling you.

**Left off on purpose.** k-nearest neighbors has no coefficient and no sensible distance across tags. A support-vector machine does not hand back a probability the queue can threshold. A neural net is a poor default on this table: the signal that beats the structural model is sparse title text, and a linear model already handles that, with weights you can read.

## What to ship

Ship the title-word score as the ranker for the 5% queue. On the test window that queue is about 9,400 questions. At 69.5% precision it contains about 6,500 questions that really go unanswered, against about 3,900 if the slots were random.

Do not ship gradient boosting over logistic regression for the structural features. The lift is about two points of precision, and logistic regression is the one you can explain. Do keep the boosting result in your pocket: the interactions and the bend were checked, and they are small next to the words in the title.

Do not ship a weekend change. Do take the code-block association to an experiment rather than to a forced wizard. The observational gap survived the edit-marker check and the closure check, and it is large enough to matter. It is still not the effect of requiring code.

## The experiment

In JavaScript, Java, C#, PHP, and Android, randomize **users**, not questions. One person posts many questions, so question-level randomization is not an independent sample. The treatment composer requires a code block or an explicit "no code needed."

Primary metric: the 24-hour answer rate from someone else. In the test window those five tags sit at a 62.2% baseline, lower than the all-tag history, which is what you want for planning because it is recent. Minimum effect worth detecting: 2 percentage points. Two-sided 5% test, 80% power. Ignoring clustering, that is about 9,100 questions per arm.

Users who ask more than once have an intraclass correlation of about 0.10 on the training window (56,000 users with at least two sampled questions). In a 14-day slice of this 10% file, a user contributes about 1.05 questions. Scaled to the full site, that is about 10.5, and the design effect is about 1.95. The inflated size is about **17,800 questions per arm**, around 1,700 users per arm. Those five tags produce about 2,800 questions a day on the full site, so the sample size arrives in about 13 days. Run **14 days** anyway, so the first week of a new composer is not the whole result.

Guardrail: questions successfully posted per randomized user. Do not ship the wizard if the treatment arm posts more than 5% fewer questions and the 95% interval for that drop excludes zero. A harsher composer can "win" the answer rate by scaring off the questions that would have failed.

The ICC was measured on the thinned sample and then applied at the scaled cluster size. If the true correlation is smaller, the experiment is a bit larger than it needs to be. That is the conservative direction.

## What an interviewer will ask

**Why 24 hours, not "ever"?** 82.7% are eventually answered by someone else, and 72.4% within 24 hours. The product failure is the wait, not only the never. The median successful answer takes 21 minutes, so 24 hours is a late cutoff, which is what you want for a label that means "this one is stuck."

**Is the code-block result causal?** No. Askers who include code may also write clearer questions, and some code is added after the first comment. The edit-marker check does not remove the gap. Only the wizard experiment estimates the effect of requiring code.

**Why is the p-value not the result?** At this sample size a 0.6-point weekend gap is "significant" and useless. The interval and the 2-point bar are the result. The weekend test is the example.

**Why did boosting barely beat logistic regression?** Because the bend in title length is a flattening, not a reversal, and the interactions do not move the top of the queue much. The forest, which attacks variance rather than bias, lands in the same place. The information the structural models are missing is the words.

**Why is year unimportant in the permutation plot and important in the chart?** The test window is one year. Shuffling a constant does nothing. Across 2008–2016 the answer rate falls by more than 30 points, and the base-rate model's Brier score gets worse on the later window because of that drift. Retrain, or at least recalibrate, as the base rate moves.

**What would you tell the product manager?** Rank the review queue with the title. Do not change the weekend rules. Test a code prompt on the five largest tags, with a guardrail on how many questions still get posted, and do not call the 3.5-point gap a causal effect until that test is done.

## Resume lines

- Built a 24-hour "answered by someone else" label on 1.26 million Stack Overflow questions and showed the rate fell from 92% in 2008 to 59% in 2016, using only features knowable at post time.
- Estimated a code-block gap of 3.5 percentage points (95% CI 3.3 to 3.7) within tag and year, rejected a weekend effect against a 2-point bar, and showed that longer titles are answered less often, with the decline flattening.
- Chose a title-word logistic model over gradient boosting on a later time window: 69.5% precision in a 5% review queue versus 60.8% for boosting, 56.2% for a no-code rule, and a 40.9% base rate. Sized the follow-up wizard test at about 18,000 questions per arm, randomized by user.
