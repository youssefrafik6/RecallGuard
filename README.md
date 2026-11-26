# RecallGuard

This repository preserves the original Southern Connecticut State University Data Science capstone and its historical results. [RecallGuard V2](https://github.com/youssefrafik6/RecallGuard-V2) extends the project with candidate review, grouped and chronological evaluation, and versioned model serving. The original model scores historical associations on the capstone sample. Its scores should not be interpreted as calibrated recall probabilities.

RecallGuard links U.S. Consumer Product Safety Commission incident reports to official recall listings using product, manufacturer and text evidence. The capstone used those historical links to compare classification models and explore differences by product category.

## Project Goal

The goal of RecallGuard is to identify products that look similar to products that were eventually recalled, based on complaint text and product details. The system is designed as an early-warning / triage tool, not as a replacement for official recall decisions.

## Key Features

- Custom linkage pipeline to connect incident reports with recall listings
- Rebuilt supervised training dataset from messy public data
- Text modeling with TF-IDF and character n-grams
- Structured feature engineering with brand, manufacturer, category, and product type
- Model comparison across:
  - Logistic Regression
  - Linear SVM
  - Complement Naive Bayes
  - XGBoost
- Threshold tuning and SMOTE experiments
- Category-specific modeling
- Streamlit app for:
  - dataset insights
  - category insights
  - risk scoring
  - model details

## Main Results

- Best overall general model: **Linear SVM**
- Best general-model F1: **0.688**
- Strongest deeper result: **Category-specific modeling**
- Best category-specific result:
  - **Home Maintenance and Structures**
  - **F1 = 0.784**

## Data Sources

This project uses public data from the U.S. Consumer Product Safety Commission (CPSC):

- **Incident Reports** from SaferProducts.gov
- **Recall Listings** from CPSC recalls data

## Project Pipeline

1. Start with two raw public CPSC files
2. Normalize product-related fields
3. Match incident reports to recall listings using multiple signals
4. Assign confidence levels to matches
5. Build a labeled training dataset
6. Perform EDA and text analysis
7. Engineer text, structured, and numeric features
8. Train and compare models
9. Extend the project into a risk-assessment app

## Streamlit App

The Streamlit app includes:

- **Dataset Insights**
- **Category Insights**
- **Risk Explorer / Risk Scorer**
- **Model Details**


## Repository Structure

```text
RecallGuard/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── app.py                                 # Existing Streamlit launch entry point
├── app/
│   └── streamlit_app.py
├── data/
│   └── training_data.csv
├── presentations/
│   └── RecallGuard_Capstone_Project.pdf
└── screenshots/
    └── capstone-dashboard.pdf
```

Run from the repository root:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

You can also launch the application directly with `streamlit run app/streamlit_app.py`.
