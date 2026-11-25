# Limitations

- Proxy labels inherit candidate selection and linkage errors. Unmatched historical rows can contain false negatives and accepted high-confidence links still require adjudication.
- The tracked source is a sampled labeled dataset with 11.1% positives. Its prevalence and calibrated scores do not represent the complete CPSC complaint population.
- Product groups use heuristic identities. Manufacturers overlap across holdouts and aliases and product variants can elude grouping.
- Model inputs may reflect edited or post-announcement narratives. Publication lags and label availability matter for prospective evaluation. The chronological benchmark measures report-date transfer on historical labels.
- Chronological performance was weak on the held-out period. Results should not be used as hazard, recall probability, or regulatory decision guidance.
- Category results are descriptive slices. Low-support categories, confidence intervals, geographic and manufacturer misclassification, and publication delays need deeper analysis.
- Coefficients and contributions describe model behavior. Correlations with complaint wording, category and brand can reflect data collection and linkage.
