# Retrospective evaluation

Run `python app/evaluation.py --audit-only` to reproduce the overlap audit, or install `requirements-evaluation.txt` and run `python app/evaluation.py` for both model comparisons. The bundled CSV has SHA-256 `ce113b9294941e6aefb89649ab4e55c76809ac342d531b192cc9b52d155c8470`.

## Selection and holdout protocol

The original dashboard ranked candidates using test-set F1. It also used the full sample for category keyword models. The revised protocol preserves the original 80/20 stratified split (seed 42) for comparison and adds a separate grouped split. Each experiment fixes its holdout before fitting models. The grouped experiment uses the first fold of a five-way shuffled stratified group split with seed 42, followed by five group-disjoint development folds. The random experiment uses five stratified development folds.

Within each development fold, TF-IDF, one-hot encoding and numeric scaling are fitted on the training rows. SMOTE candidates resample only transformed training rows and reuse those samples within the fold. The candidates use sparse features, and XGBoost uses fold-local balanced sample weights. Pooled out-of-fold F1 selects the hyperparameters and classifier. For the tuned logistic regression candidate, selection also searches cutoffs from 0.10 to 0.90 in steps of 0.05. Ties prefer 0.50 within the cutoff search and the first candidate in a fixed order elsewhere. The selected model is refitted on all development rows and scored once on its holdout using its fixed decision rule. CV scores participate in selection and may be optimistic. Dashboard category metrics use the scorer’s logistic regression development predictions and keyword models are fitted on development rows.

## Related-report audit

The dataset has 3,780 reports, 64 columns, 420 positive labels, no identical full records and no repeated report numbers. Report number identifies an incident. The CSV has no universal product ID. The grouped split uses connected components from repeated report numbers, normalized brand or manufacturer scoped to the model, UPC strings of plausible lengths, shared `recall_index`, and exact or highly similar incident descriptions. Empty or recognized placeholder identifiers are skipped. Brand or manufacturer alone never joins reports. Product codes describe product classes.

An exact text match uses case and punctuation normalization while retaining numbers. The near-duplicate audit compares binary word-trigram cosine similarity at 0.90 or higher, using descriptions with at least 12 tokens and counting nonidentical normalized texts. It found one exact text pair and two near pairs in the dataset. One near pair crossed the random boundary. This heuristic can miss paraphrases and short duplicates and can connect reports sharing boilerplate.

| Match rule | Random shared keys or pairs | Random holdout rows touched | Grouped shared keys or pairs | Grouped holdout rows touched |
| --- | ---: | ---: | ---: | ---: |
| Manufacturer | 187 | 582 | 162 | 565 |
| Brand | 148 | 405 | 143 | 421 |
| Manufacturer + model | 59 | 73 | 0 | 0 |
| Brand + model | 44 | 54 | 0 | 0 |
| UPC | 3 | 4 | 0 | 0 |
| Recall candidate | 50 | 106 | 0 | 0 |
| Near incident description | 1 | 1 | 0 | 0 |
| Exact normalized incident description | 0 | 0 | 0 | 0 |
| Report number | 0 | 0 | 0 | 0 |

Rows can match multiple rules. Eligible UPCs were available on 269 reports. The recall index identifies a candidate from the original linkage stage and is present for 724 reports, including 304 with unlinked labels. Campaign-level grouping uses these candidates conservatively even where the linkage label is negative. Candidate IDs and other linkage metadata are excluded from model features. The group rules produce 3,220 components, with 79 reports in the largest. Inconsistent model strings and missing identifiers can split related products. Reused model names, noisy candidate links, and transitive matches can combine distinct products. Grouping also permits manufacturer and brand overlap.

The random holdout contains 84 positives among 756 rows (11.11%). The grouped holdout contains 83 positives among 755 rows (10.99%). The corresponding development sets contain 336 positives among 3,024 random rows and 337 among 3,025 grouped rows. The five grouped validation folds had 604–606 rows and 67–68 positives each.

## Results and limitations

The research presentation reports historical F1 of **0.688** for the general Linear SVM and **0.784** for Home Maintenance and Structures. These are original capstone findings.

The following run used Python 3.11.13. The [machine-readable benchmark](data/evaluation_results.json) records the library versions, all development scores, and the holdout confusion matrices. The full run took about 14 minutes on the verification machine. Model Details reads the saved benchmark.

| Split | Selected classifier | Tuned cutoff | Development CV F1 | Holdout accuracy | Holdout precision | Holdout recall | Holdout F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original random | Linear SVM | Default | 0.6598 | 0.9339 | 0.7429 | 0.6190 | 0.6753 |
| Related-report grouped | Logistic Regression (tuned threshold) | 0.10 | 0.2032 | 0.6861 | 0.2467 | 0.9036 | 0.3876 |

The splits contain different reports and selected configurations can differ. The performance gap measures sensitivity to the split design without isolating the effect of any single source of overlap.

Both revised experiments reuse historical capstone data. The original random holdout already informed capstone development, and the grouped holdout is a new partition of the same sample. Neither can establish prospective performance. The groups and labels depend on incomplete linkage fields, and some structured features are also evidence used to construct the labels. Complaint text can mention a recall that was already public. The original sampled class proportions, label errors, and these retrospective cues limit interpretation. Future-recall estimates would require independent, time-appropriate data and separate calibration.
