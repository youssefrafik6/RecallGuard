# Model bundles

`recallguard train --design group` generates `results/runs/group.joblib` plus JSON metadata. The bundle contains the fitted encoders, vectorizers, estimator, sigmoid calibrator, model thresholds, label mappings, input schema, evaluation and dependency versions. Set `RECALLGUARD_MODEL` to a trusted bundle you generated. Generate the artifact within the serving environment. Joblib deserialization can execute code and must never receive external files.
