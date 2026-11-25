# Model evaluation

The benchmark uses all 3,780 observations in the tracked capstone CSV (420 positives). Its SHA-256 is recorded in the [machine-readable benchmark](../results/metrics/historical_proxy_benchmark.json) alongside package versions, seed 42, partition sizes, individual variants and calibration metrics. The CLI regenerates the full ROC, PR and validation threshold curves. The historical results are reported separately in the [archived README](CAPSTONE_README.md).

| Design | Selected by validation AP | Test rows / positive rows | Test average precision | Test F1 | Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| Random stratified | Logistic regression | 756 / 84 | 0.551 | 0.562 | 0.068 |
| Product-grouped | Logistic regression | 755 / 83 | 0.412 | 0.402 | 0.082 |
| Chronological, group-purged | XGBoost | 756 / 97 | 0.144 | 0.019 | 0.112 |

The chronological precision-recall baseline is the positive share of 0.128. At its frozen validation threshold, XGBoost flagged ten rows and recovered one of 97 positives. Its top 25 contained one positive. The chronological validation-to-test transfer was poor. The group run returned 17 positives in its top 25. Each model family selects its variant using validation AP. The reported winner is the best nondummy family under that same rule.

| Model | Random F1 / AP | Grouped F1 / AP | Chronological F1 / AP |
| --- | ---: | ---: | ---: |
| Dummy prior | 0.200 / 0.111 | 0.198 / 0.110 | 0.227 / 0.128 |
| Logistic regression | 0.562 / 0.551 | 0.402 / 0.412 | 0.046 / 0.134 |
| Linear SVM | 0.556 / 0.529 | 0.434 / 0.422 | 0.045 / 0.128 |
| Complement NB | 0.525 / 0.564 | 0.374 / 0.372 | 0.019 / 0.130 |
| XGBoost | 0.417 / 0.431 | 0.293 / 0.298 | 0.019 / 0.144 |

AP measures ranking among rare positives. F1 gives equal weight to positive-class precision and recall. The dummy's tuned threshold labels all rows positive. Its rank-based top-K contains score ties broken by split order. Reported figures are descriptive for one held-out split per design and have no uncertainty interval.

Word and character TF-IDF, categorical encoders and a scaled description-length feature are fitted once on training only and reused across candidates. Logistic regression compares class weights, regularization and seeded random oversampling. Linear SVM compares regularization and class weights. XGBoost compares training positive weights. Complement NB sweeps smoothing. Random oversampling duplicates training positives only. Historical sparse-text SMOTE remains documented in the original code history. Category metrics are slices with at least 20 test examples and both classes.

A regularized sigmoid is fitted on the independent calibration split. A negative fitted slope is replaced by a constant calibration model. Brier score, log loss and quantile reliability points are evaluated on test. The high-priority boundary maximizes proxy F1 on validation. Moderate starts at the lower of that boundary and the 75th percentile validation score. When the boundaries coincide, the moderate band is empty. These values express relative review priority under the benchmark's sample distribution.

## F1-focused product-grouped follow-up

The original product-grouped split (seed 42, 755 test reports, 83 positives) was reused for an exploratory comparison. The new selection option ranks model variants by their best validation F1, with validation AP breaking ties. The threshold is chosen on validation after independent sigmoid calibration. The fielded representation fits product name word and character n-grams separately from incident narrative word n-grams, retaining the existing categorical and numeric features.

| Configuration | Held-out F1 | Precision | Recall | AP | TP | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original combined text, AP selection | 0.402 | 0.407 | 0.398 | 0.412 | 33 | 48 |
| Combined text, F1 selection | 0.386 | 0.452 | 0.337 | 0.393 | 28 | 34 |
| Separate product and narrative text, F1 selection | 0.455 | 0.532 | 0.398 | 0.400 | 33 | 29 |

The fielded run selected unweighted logistic regression with `C=1.0`. Its validation F1 was 0.519 and its frozen calibrated threshold was 0.2295. Its 0.053 absolute F1 gain reflects fewer false positives at the same recall. The AP decline shows the gain is specific to the operating point. Because the holdout was previously inspected, this result needs confirmation on a fresh grouped sample. This experiment did not rerun chronological evaluation.

```bash
recallguard train --design group --selection f1 --out results/runs/f1-group
recallguard train --design group --selection f1 --feature-set fielded \
  --out results/runs/f1-fielded
RECALLGUARD_MODEL=results/runs/f1-fielded/group.joblib streamlit run app.py
```

## Chronological generalization follow-up

Both runs use the previously inspected January 19, 2023 to December 30, 2025 holdout of 756 reports, including 97 positive proxy labels. This is a retrospective replay with a chronologically later test set. No test labels or test scores selected a family, feature set, weight or operating threshold. Across four TF-IDF profiles, estimators and weights were fit on training only, and selected by maximum validation F1 with average precision as the tie-breaker. The four profiles were combined word and character features with structured fields, combined text only, word-only TF-IDF, and separate product and narrative TF-IDF with structured fields. A distinct, later pre-test partition fitted a sigmoid to each selected family. Validation fixed the operating threshold and test evaluation was performed after choices were frozen.

The stricter primary run additionally groups matching historical recall indices, checksum-equivalent GTINs, and manufacturer/model pairs with punctuation differences. Historical indices include tentative links on some legacy negatives and are used only to purge, never as scoring features. In the original proxy grouping 32 test positives shared a campaign with an earlier positive, and 83 test rows shared any archived campaign candidate with an earlier row. The strict run reduced both counts to zero, preserved the test cohort, and excluded 476 rows overall versus 269 in the earlier split. Remaining product aliases or identity errors can still evade grouping.

| Strict chronological model | Validation F1 | Precision | Recall | F1 | AP | P@10 | P@25 | P@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy prior | 0.094 | 0.128 | 1.000 | 0.227 | 0.128 | 0.10 | 0.08 | 0.10 |
| **Weighted Logistic Regression, separate text fields** | **0.235** | **0.115** | **0.062** | **0.081** | **0.123** | **0.10** | **0.08** | **0.10** |
| Weighted Linear SVM, combined text and structured features | 0.234 | 0.136 | 0.082 | 0.103 | 0.114 | 0.00 | 0.08 | 0.14 |
| Complement Naive Bayes, combined text and structured features | 0.140 | 0.134 | 0.412 | 0.202 | 0.122 | 0.10 | 0.12 | 0.08 |
| Weighted XGBoost, word-only TF-IDF | 0.168 | 0.145 | 0.433 | 0.218 | 0.152 | 0.10 | 0.16 | 0.18 |

Bold marks the model selected before test. XGBoost's higher observed test F1 came from flagging 289 reports and recovering 42 positives. It remained below the flag-everything dummy on F1. The dummy has tied scores, so its observed precision at K depends on tie order. For a random review budget, expected precision is the test prevalence of 0.128.

On the original proxy-only split, the expanded search selected a fielded SVM using validation F1. Its test F1 was 0.035, AP was 0.125, precision was 0.125, recall was 0.021 and P@25 was 0.12. The original AP-selected XGBoost result was 0.019 F1 with 0.144 AP. The strict Logistic Regression model reached F1 0.081 by recovering six positives among 52 flags. Its P@25 was two positives in 25 and its AP was below the prevalence baseline. Its validation-selected threshold was 0.1235. The default calibrated cutoff of 0.5 flagged no positives. Its Brier score was 0.1144 versus 0.1134 for the constant dummy, so calibration gave interpretable proxy scores without evidence of a calibration gain on the test period.

### Why transfer was weak

- **Temporal shift.** In the stricter replay, 424 of 756 test rows had a nonblank normalized manufacturer unseen in training. Manufacturer total variation was 0.761 on a zero-to-one scale, versus 0.097 for product category. About 6.4% of test token occurrences were unseen in training. TF-IDF mainly learns historical names and wording, and the feature allowlist deliberately lacks raw part identifiers and later recall information.
- **Class imbalance and campaign concentration.** The strict split contained 23/1,269 training positives (1.8%), 31/626 validation positives (5.0%), 58/653 calibration positives (8.9%) and 97/756 test positives (12.8%). The training positives represented 20 historical campaigns. Reweighting changes the relative training loss and cannot replace unseen positive examples. These are observed shares in a sampled retrospective dataset.
- **Label and linkage differences.** Eleven of 23 strict training positives were historically medium confidence, versus 16 of 97 test positives. The 365-day identifier-aware reconstruction agreed on 39 of the 97 historical test positives and yielded no new high-confidence positives among the other test rows. The rules and outcome windows differ, and neither method supplies independently reviewed truth. This comparison cannot estimate false-negative or false-positive rates.
- **Missing identity and availability evidence.** Among 756 test reports, 530 had a blank UPC and 103 a blank model field. The median filing-to-publication delay was 20 days. Raw recall dates previously showed some reconstructed legacy links were already announced when the incident went public. File-date ordering alone cannot verify that the feature values and labels would have been available at each forecast cutoff.

### Defensible use and next evaluation

Use the linker as a **review queue for associating incident reports with already published recall records**. Show the evidence and competing matches, and let reviewers confirm or reject them. Evaluate that workflow on a blinded, adjudicated sample including ambiguous matches and unmatched incidents. Measure precision at the team's review budget and reviewer time. For a future-recall model, collect point-in-time public report snapshots, define a fixed horizon from publication, wait until outcomes mature, and evaluate on a new cohort with rolling cutoffs and product/campaign purging.

```bash
recallguard audit-temporal --group-campaigns \
  --linked-observations data/processed/observations.csv
recallguard train --design chronological --feature-set compare --selection f1 \
  --group-campaigns --out results/runs/temporal-strict
```

Omit `--group-campaigns` to replay the original product-proxy grouping. The aggregate report includes both configurations and the validation candidate metrics. [Machine-readable temporal benchmark](../results/metrics/temporal_generalization_benchmark.json)
