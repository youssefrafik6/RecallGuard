# RecallGuard

RecallGuard explores historical associations between consumer incident reports and U.S. Consumer Product Safety Commission (CPSC) recalls. The Streamlit app charts the labeled dataset, compares text classifiers, and scores individual reports.

## Data and methodology

The [bundled CSV](data/training_data.csv) contains 3,780 labeled incident reports, including 420 positive labels. The historical linkage study used CPSC consumer incident reports and recall listings, which lack a shared product identifier. The app consumes the labeled CSV. The [research presentation](presentations/RecallGuard_Capstone_Project.pdf) documents label construction and credits Youssef Rafik, Southern Connecticut State University, Spring 2026.

Python, pandas and scikit-learn handle data preparation, TF-IDF features and model evaluation. Experiments include XGBoost, SMOTE via imbalanced-learn, and category-specific classifiers. Streamlit and Plotly present the results.

## Getting started

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The application fits models on first load and caches the result. Open **Dataset Insights** or **Category Insights** to explore the labels, **Model Details** to compare metrics, or **Risk Scorer** to enter product and incident details. The scorer combines a classifier output with a rule-based hazard severity signal.

## Historical results and limitations

The research presentation reports an overall F1 of 0.688 for Linear SVM and a category-specific F1 of 0.784 for Home Maintenance and Structures. These are historical experiment results. The score reflects the dataset’s linkage labels and the heuristic severity rules. Independent prospective validation would be required to estimate future recall probability.

[Dashboard walkthrough](screenshots/capstone-dashboard.pdf) · [MIT License](LICENSE)
