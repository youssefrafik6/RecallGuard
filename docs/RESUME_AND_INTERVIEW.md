# Resume and interview notes

## Two-line resume description

RecallGuard links CPSC consumer incident reports to official recall listings and ranks retrospective recall associations with sparse NLP models. The capstone's portfolio upgrade adds auditable linkage, product-grouped and chronological evaluations, and an artifact-backed Streamlit and FastAPI demo.

## Resume bullets

- Built a bounded CPSC record-linkage workflow that audited 1.14 million candidate pairs from 65,058 incident reports and 9,623 validated recall listings. Added model and UPC matching that recovered 204 of 207 date-eligible pairs in a historical audit.
- Benchmarked sparse TF-IDF classifiers and imbalance strategies on 3,780 historical observations. Quantified how held-out performance changed when products were separated and a later reporting period was held out.
- Packaged a reproducible training CLI and shared inference layer for Streamlit and FastAPI, with pytest coverage, Docker, and a GitHub Actions workflow.

These bullets describe the resulting repository. In interviews, claim the work you personally performed and describe the portfolio upgrade as a later collaboration where applicable. Your original capstone designed the research question, historical linkage and labeling approach, benchmark experiments, and original Streamlit app.

## 30-second explanation

I started RecallGuard as a capstone to see whether consumer complaints could help identify products associated with CPSC recalls. The two sources lacked a shared product identifier, so I built a labeled sample by linking product evidence across the files. The portfolio upgrade records why candidates match, evaluates the model with product and time separation, and serves a trained model through a dashboard and API. Performance fell sharply on newer reports, which is why I describe it as retrospective association analysis.

## Two-minute walkthrough

Start with the supervision problem. Consumer reports describe an incident, while a recall listing describes a product and announcement. Product evidence such as brand, manufacturer and model identifiers is inconsistently present. The capstone used normalization, fingerprints and multiple match signals to build the historical labels. The reconstructed linker uses bounded token blocks and explicit date and score gates. Its strongest labels are still proxies and unresolved rows remain blank.

Next explain the benchmark. Combine word and character TF-IDF with optional one-hot product fields and text length. Fit transformers only on training rows. Compare dummy, logistic, SVM, Complement NB and XGBoost. Use validation to select hyperparameters and threshold and reserve a calibration partition for sigmoid fitting. Evaluate on three held-out designs. The group split reduced F1 from 0.562 to 0.402 for the validation-selected logistic model. The chronological design yielded 0.019 F1 for validation-selected XGBoost. Different data slices, positive shares and time shift complicate attribution.

Finish with deployment. The CLI writes a fitted pipeline, calibrator, configuration and metrics. Streamlit loads the trusted artifact once and shows data, scoring and diagnostics. FastAPI shares validation and inference. Docker verifies runtime packaging locally. The next research step is adjudicating candidate links and obtaining point-in-time report snapshots.

## Architecture to draw on a whiteboard

Draw two inputs into the linkage stage and two outputs from it, a candidate evidence table and observations with nullable labels. Draw the historical labeled snapshot into separate train, validation, calibration, and test partitions. Train fits sparse preprocessing and estimators. Validation selects hypotheses and the cutoff. Calibration maps base scores to the sample proxy scale. Test measures the frozen procedure. A serialized estimator plus schema, thresholds and metadata feeds the dashboard and API.

## Ten interview questions

1. **Why did you need record linkage?** Incident and recall rows lack a shared universal product identifier. Candidate blocking narrows pairs, and multiple product signals provide auditable evidence of identity.
2. **How did you construct labels?** The capstone retained high and selected medium-confidence positives and used unlinked observations as comparison rows. The reconstruction labels a unique, date-eligible high candidate and leaves all other rows unresolved.
3. **How did you find or prevent leakage?** I reviewed split membership and the feature allowlist. Product-proxy groups stay together and earlier duplicates are purged for chronological testing. I would require as-of snapshots to eliminate possible hindsight embedded in raw text.
4. **Why did class imbalance matter?** Only 420 of 3,780 historical rows are positive. A classifier can achieve high accuracy by predicting the majority while surfacing no useful cases.
5. **Why F1 and average precision?** F1 measures precision and recall at an operating point. AP measures ranking across recall levels. Precision at K reflects a fixed review budget.
6. **How did you tune the threshold?** Fit the sigmoid on calibration rows, transform validation scores, and pick the F1-maximizing validation cutoff. Freeze it for the final holdout. Different review costs could justify another rule.
7. **Why might Linear SVM work well?** Sparse text separates examples in a high-dimensional space, where a linear margin can generalize efficiently. The original F1 of 0.688 is historical. The newer validation-selected winner depends on the split.
8. **How does Streamlit work?** It caches the local dataset and trained artifact, exposes exploration and held-out metrics, and calls shared inference for assessment. Artifact-free mode supports dataset and linkage exploration.
9. **How does FastAPI work?** It loads an operator-managed model at startup, validates typed incident fields, and returns the same priority and contributions as the app. It distinguishes malformed requests from an unavailable model.
10. **What would you improve next?** Create a stratified human-adjudicated linkage set, capture first-available report snapshots, define an outcome horizon with censoring, and run rolling prospective holdouts with confidence intervals.
