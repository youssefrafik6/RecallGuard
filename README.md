## What makes this project different?

A major part of this project was not just training a classifier, but building a usable supervised dataset from two messy public CPSC files that could not be directly joined. That data-linkage and label-construction step was the foundation for all later modeling results.

# RecallGuard

RecallGuard is a Data Science capstone project that estimates product recall risk from consumer safety complaints.

The project starts with two public datasets from the U.S. Consumer Product Safety Commission (CPSC): consumer incident reports and official recall listings. Because these files do not share a clean product ID, I built a custom linkage pipeline to connect them and create a supervised training dataset. I then trained and compared multiple machine learning models, performed category-specific analysis, and built a Streamlit app that turns the project into a practical risk-assessment tool.

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

## Why This Project Matters

Public complaint data may contain warning signs before a recall is officially announced, but the data is too large and too messy to review manually. RecallGuard shows how public safety complaint data can be turned into a practical machine learning workflow for product risk triage.

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

It demonstrates how the project can be used as a practical product risk-assessment tool.

## Repository Structure

```text
.
├── app.py
├── training_data.csv
├── requirements.txt
├── README.md
├── screenshots/
└── slides/
