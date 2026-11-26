"""Shared data loading and feature preparation for training and scoring."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

from risk import HAZARD_WORDS, clean_text, keyword_count

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "training_data.csv"

CUSTOM_STOPWORDS = {
    "said", "would", "could", "also", "one", "two", "get", "got",
    "still", "back", "even", "like", "product", "incident",
    "description", "report", "submitted", "submitter", "amazon",
    "seller", "page", "business", "item",
}

# Combine sklearn's stopwords with our custom ones
ALL_STOPWORDS = ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS)

# The columns we need from the dataset for modeling
REQUIRED_COLS = [
    "label_recalled",
    "Product Description",
    "Incident Description",
    "Manufacturer / Importer / Private Labeler Name",
    "Brand",
    "Model Name or Number",
    "Product Category",
    "Product Sub Category",
    "Product Type",
]



FEATURE_COLS = REQUIRED_COLS[1:]
NUMERIC_COLS = [
    "product_word_count", "incident_word_count", "product_hazard_count",
    "incident_hazard_count", "has_model_info",
]


def load_data(path=DATA_PATH):
    # Read identifiers as strings to preserve leading zeroes.
    frame = pd.read_csv(path, low_memory=False, dtype={"UPC_norm": "string"})
    missing = set(REQUIRED_COLS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if frame["label_recalled"].isna().any() or set(frame["label_recalled"]) != {0, 1}:
        raise ValueError("label_recalled must contain both binary classes and no missing values.")
    return frame


def prepare_features(frame):
    features = frame[FEATURE_COLS].copy()
    for column in FEATURE_COLS:
        features[column] = features[column].astype("string").fillna("").str.strip()
    for name, column in zip(("product", "incident"), FEATURE_COLS[:2]):
        features[f"{name}_word_count"] = features[column].map(
            lambda text: len(clean_text(text).split())
        )
        features[f"{name}_hazard_count"] = features[column].map(
            lambda text: keyword_count(text, HAZARD_WORDS)
        )
    features["has_model_info"] = (
        ~features["Model Name or Number"].str.lower().isin(["", "unknown"])
    ).astype(int)
    features[FEATURE_COLS] = features[FEATURE_COLS].replace("", "unknown")
    return features


def prepare_prediction(values):
    descriptions = (values.get(column) for column in FEATURE_COLS[:2])
    if not any(isinstance(text, str) and text.strip() for text in descriptions):
        raise ValueError("Please enter at least a product description or incident text.")
    return prepare_features(pd.DataFrame([values], columns=FEATURE_COLS))


def build_preprocessor(numeric_cols=NUMERIC_COLS):
    """
    Creates the preprocessing pipeline that transforms raw columns
    into model-ready features.

    It applies:
    - TF-IDF (word-level, bigrams) on Product Description
    - TF-IDF (word-level, bigrams) on Incident Description
    - TF-IDF (character n-grams) on Incident Description
    - One-hot encoding on Manufacturer, Brand, Model, Category,
      Sub Category, and Product Type
    - MaxAbsScaler on the numeric features
    """
    preprocessor = ColumnTransformer(
        transformers=[
            # Word-level TF-IDF on product description
            ("product_word_tfidf", TfidfVectorizer(
                stop_words=list(ALL_STOPWORDS),
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                max_features=5000,
                sublinear_tf=True,
            ), "Product Description"),

            # Word-level TF-IDF on incident description
            ("incident_word_tfidf", TfidfVectorizer(
                stop_words=list(ALL_STOPWORDS),
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                max_features=5000,
                sublinear_tf=True,
            ), "Incident Description"),

            # Character-level TF-IDF on incident description
            # This captures patterns like partial words and typos
            ("incident_char_tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=4000,
                sublinear_tf=True,
            ), "Incident Description"),

            # One-hot encode the categorical fields
            ("manufacturer_ohe", OneHotEncoder(handle_unknown="ignore"),
             ["Manufacturer / Importer / Private Labeler Name"]),
            ("brand_ohe", OneHotEncoder(handle_unknown="ignore"),
             ["Brand"]),
            ("model_ohe", OneHotEncoder(handle_unknown="ignore"),
             ["Model Name or Number"]),
            ("category_ohe", OneHotEncoder(handle_unknown="ignore"),
             ["Product Category"]),
            ("subcategory_ohe", OneHotEncoder(handle_unknown="ignore"),
             ["Product Sub Category"]),
            ("ptype_ohe", OneHotEncoder(handle_unknown="ignore"),
             ["Product Type"]),

            # Scale numeric features (sparse-friendly scaling)
            ("numeric_scaled", MaxAbsScaler(), numeric_cols),
        ],
        remainder="drop",
    )
    return preprocessor
