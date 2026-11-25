# Repository audit

Baseline checkout: `ef0feb2`. Git history contains the Streamlit source, an already-labeled CSV, a capstone slide deck, screenshots, and short-lived raw-directory placeholders. There are no tracked notebooks, raw source exports, linkage implementation, fitted estimators, locked environment, CI, or tests at this revision. The historical README is preserved in full.

| Historical README claim | Classification | Evidence |
| --- | --- | --- |
| CPSC incident and recall sources | Supported only by historical documentation | The checked-in artifact has CPSC-style incident fields. Local CPSC exports exist, but their exact relationship to this CSV is unverified. |
| Custom normalization, candidate linkage, confidence and supervised dataset reconstruction | Not currently reproducible | CSV contains normalized values, match rules, match signals, confidence and binary labels. Git history has no original builder. |
| Text TF-IDF, char n-grams and structured fields | Verified from source code | `build_preprocessor` fits word TF-IDF on descriptions, character TF-IDF on incidents, one-hot product fields and scaled count features. |
| Logistic regression, Linear SVM, Complement NB and XGBoost comparison | Verified from source code | `train_all_models` defines these alongside random forest and SGD. |
| Training-side threshold tuning and SMOTE | Verified from source code | Out-of-fold LR scores select an F1 cutoff. SMOTE operates on dense transformed training rows. |
| Category analysis | Verified from source code | Category metrics slice general LR test predictions. Separate full-subset text fits generate descriptive keywords. |
| Linear SVM F1 0.688 and Home Maintenance and Structures F1 0.784 | Supported only by historical documentation | Historical README values. Exact run inputs, split membership, metrics and environment are absent. The current code does not establish that the second metric came from an independently evaluated category-trained model. |
| Dataset insights, category insights, scoring and model details | Verified from source code | The Streamlit monolith defines those views and trains models in process. |
| Validated early warning | Incorrect or unsupported | The application uses a random row split and mixes a model output with hand-authored severity heuristics. |

CSV inspection verified by execution: 3,780 parsed observations (420 linked, 3,360 other). Physical line count exceeds observation count because narratives contain embedded newlines. Workspace also contains an earlier 3,271-row labeled dataset (416 positives), 416-row positive-match audit and an exploratory notebook using an absolute Colab path. These are separate artifacts with different schemas. Desktop holds a source snapshot, raw exports and presentation variants. Original raw snapshots stay local.

The original app trains on startup, densifies sparse arrays, shares and refits a mutable preprocessor between model branches, selects a winning model with test F1, and serves LR independently of that winner. The displayed composite is an adaptive blend with severity keywords and hand-selected 0.25/0.55 bands. It has no established probabilistic interpretation. The original CSV includes product identifiers, free text, victim and geographic fields. Retain research history while reviewing redistribution and minimizing derived exports.

The archived deck details model-like fingerprints, a second linkage pass, and headline metrics on pages 6–9 and 14–17. Its label recipe emphasizes retaining high-confidence positives, while the checked-in data contains 330 high and 90 medium positives. The screenshot PDF documents the historical interface. Source and slide claims of calibrated LR probabilities are unsupported because the application provides no calibration analysis.

The Desktop app snapshot is byte-identical to the original tracked application. The original application remains retrievable with `git show ef0feb2:app.py`. No historical commit or license text was rewritten.

A narrow signature search of textual Git history covered common private-key headers and AWS, GitHub, and Stripe token formats. A separate scan of the historical CSV found 164 email-shaped strings and 893 US-phone-shaped strings, including possible false matches and repeats. This was a pattern scan, so it did not identify individuals or certify redaction. The CSV and any local image that embeds it require privacy review before distribution.
