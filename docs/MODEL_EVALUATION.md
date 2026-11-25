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
