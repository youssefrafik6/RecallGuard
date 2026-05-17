import os
import re
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# sklearn imports for modeling
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict,
)
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# XGBoost for gradient boosted trees
from xgboost import XGBClassifier

# SMOTE for oversampling the minority class
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# Suppress sklearn convergence warnings so the app looks clean
warnings.filterwarnings("ignore")


# SECTION 2: STREAMLIT PAGE CONFIGURATION
# This sets the browser tab title, icon, and layout.
st.set_page_config(
    page_title="RecallGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# SECTION 3: CUSTOM CSS STYLING
# This makes the app look polished and presentation-ready.
# I use custom fonts, dark metric cards, colored risk boxes, etc.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');

/* Apply DM Sans to main text elements */
.stApp, .stMarkdown, .stMarkdown p, .stMarkdown li,
div[data-testid="stText"], .stCaption, .stSelectbox label,
.stTextInput label, .stTextArea label, .stRadio label {
    font-family: 'DM Sans', sans-serif !important;
}

/* Use Space Mono (monospace) for headings */
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

/* Dark gradient sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stRadio label {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem;
    padding: 0.45rem 0;
}

/* Dark styled metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.6rem !important;
}

/* Risk score result boxes (green/orange/red) */
.risk-box {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    margin: 12px 0;
}
.risk-box h2 {
    margin: 0 0 4px 0;
    font-size: 2.6rem;
    font-family: 'Space Mono', monospace;
}
.risk-box p { margin: 0; font-size: 0.95rem; opacity: 0.85; }
.risk-low { background: linear-gradient(135deg, #065f46, #047857); color: #d1fae5; }
.risk-moderate { background: linear-gradient(135deg, #92400e, #b45309); color: #fef3c7; }
.risk-high { background: linear-gradient(135deg, #991b1b, #dc2626); color: #fee2e2; }

/* Hero banner on the dashboard */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border-radius: 20px;
    padding: 48px 40px;
    text-align: center;
    margin-bottom: 28px;
    border: 1px solid #334155;
    box-shadow: 0 8px 40px rgba(0,0,0,0.2);
}
.hero h1 { color: #f8fafc; font-size: 2.8rem; margin-bottom: 8px; }
.hero p { color: #94a3b8; font-size: 1.1rem; max-width: 640px; margin: 0 auto; }

/* Colored keyword chips for category insights */
.kw-chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    margin: 3px 4px;
    font-family: 'Space Mono', monospace;
    font-weight: 500;
}
.kw-pos { background: #fee2e2; color: #991b1b; }
.kw-neg { background: #d1fae5; color: #065f46; }

/* Small uppercase section labels */
.section-header {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    margin-bottom: 12px;
    font-family: 'Space Mono', monospace;
    border-bottom: 2px solid #1e293b;
    padding-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)


# SECTION 4: CONSTANTS AND WORD LISTS
# These are the fixed values used throughout the app:
# - path to the dataset
# - hazard keywords for feature engineering
# - severity word lists for the risk scorer
# - stopwords for TF-IDF
# - required column names

DATA_PATH = os.path.join(os.path.dirname(__file__), "training_data.csv")


HAZARD_WORDS = {
    "fire", "burn", "smoke", "shock", "explosion", "explode",
    "hazard", "injury", "injuries", "death", "deaths",
    "overheat", "overheating", "choking", "fall", "blade",
    "laceration", "cut", "flame", "sparks", "melt", "melted",
}

# Severity word lists for the Risk Scorer's hazard-signal scoring.
# These are split into three tiers so the risk scorer can tell
# the difference between "it broke" and "it exploded and my child
# was hospitalized." Each tier gets a different weight.

# Critical = most dangerous language (weight: 0.35 per hit)
SEVERITY_CRITICAL = {
    "death", "deaths", "died", "killed", "fatal", "fatality",
    "hospitalized", "hospital", "emergency", "er", "icu",
    "exploded", "explosion", "electrocuted", "electrocution",
    "amputation", "amputated", "suffocated", "suffocation",
    "permanent", "brain", "coma", "unconscious", "seizure",
    "third degree", "second degree",
}

# High severity = serious harm language (weight: 0.15 per hit)
SEVERITY_HIGH = {
    "fire", "burn", "burned", "burns", "smoke", "smoking",
    "flame", "flames", "ignite", "ignited", "caught fire",
    "shock", "shocked", "electrical shock", "sparks",
    "overheat", "overheating", "overheated", "melted", "melt",
    "choking", "choked", "laceration", "fracture", "fractured",
    "stitches", "bleeding", "blood", "concussion", "collapsed",
    "shattered", "broke apart", "fell apart", "came off",
    "child", "children", "infant", "baby", "toddler", "kid",
}

# Moderate severity = general concern language (weight: 0.05 per hit)
SEVERITY_MODERATE = {
    "hazard", "injury", "injuries", "injured", "hurt",
    "cut", "cuts", "bruise", "bruised", "rash", "swelling",
    "pain", "painful", "fell", "fall", "dropped", "broken",
    "defect", "defective", "malfunction", "malfunctioned",
    "recall", "recalled", "warning", "danger", "dangerous",
    "unsafe", "risk", "faulty",
}

# Custom stopwords added on top of sklearn's built-in list.
# These are common words in CPSC reports that don't help the model.
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


# SECTION 5: HELPER FUNCTIONS
# These are utility functions used in multiple places:
# - clean_text: cleans raw text for modeling
# - keyword_count: counts how many hazard words appear
# - compute_severity_score: scores how dangerous the text sounds
# - composite_risk_score: blends the ML probability with severity
# - get_metrics: computes accuracy/precision/recall/f1

def clean_text(text):
    """
    Steps:
      1. Lowercase everything
      2. Remove URLs
      3. Remove the word "redacted"
      4. Replace dimension patterns (e.g. "4 x 2 x 1") with "dimension"
      5. Replace measurement patterns (e.g. "10 inch") with "measurement"
      6. Remove long numbers (2+ digits)
      7. Keep only letters and spaces
      8. Collapse extra whitespace
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\bredacted\b", " ", text)
    text = re.sub(
        r"\b\d+(\.\d+)?\s*x\s*\d+(\.\d+)?(\s*x\s*\d+(\.\d+)?)?\b",
        " dimension ", text,
    )
    text = re.sub(
        r"\b\d+(\.\d+)?\s*(inch|inches|lb|lbs|pound|pounds|oz|ounces"
        r"|cm|mm|mah|amp|amps|volt|volts|v)\b",
        " measurement ", text,
    )
    text = re.sub(r"\b\d{2,}\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_count(text, vocab):
    """
    Counts how many words from vocab appear in the cleaned text.
    Used to create the hazard-word count features.
    """
    words = clean_text(text).split()
    count = 0
    for w in words:
        if w in vocab:
            count += 1
    return count


def compute_severity_score(text):
    """
    Computes a rule-based severity score from 0 to 1 based on how
    dangerous the text sounds. This is NOT from the ML model, it's
    a separate signal used in the Risk Scorer to catch cases where
    the incident is clearly dangerous but the model gives a low score.

    Returns:
      score: float between 0 and 1
      crit_hits: how many critical-tier words were found
      high_hits: how many high-tier words were found
      mod_hits: how many moderate-tier words were found
    """
    cleaned = clean_text(text)
    words = set(cleaned.split())

    # Build bigrams too, so we can match phrases like "caught fire"
    word_list = cleaned.split()
    if len(word_list) > 1:
        bigrams = set()
        for i in range(len(word_list) - 1):
            bigrams.add(word_list[i] + " " + word_list[i + 1])
    else:
        bigrams = set()

    # Combine single words and bigrams
    all_tokens = words | bigrams

    # Count how many words from each severity tier appear
    crit_hits = len(all_tokens & SEVERITY_CRITICAL)
    high_hits = len(all_tokens & SEVERITY_HIGH)
    mod_hits = len(all_tokens & SEVERITY_MODERATE)

    # Weighted sum: critical words count the most
    raw = crit_hits * 0.35 + high_hits * 0.15 + mod_hits * 0.05

    # Cap the score at 1.0
    score = min(1.0, raw)

    return score, crit_hits, high_hits, mod_hits


def composite_risk_score(model_proba, severity_score):
    """
    Blends the ML model's probability with the severity score
    to get a final composite risk score.

    Why do we need this?
    The ML model was trained to predict formal CPSC recalls, not danger.
    So a battery that explodes on a child might get a low model score
    because batteries in that category were rarely recalled.
    The severity score catches that gap.

    The weighting is adaptive:
    - When severity is low (0):  70% model, 30% severity
    - When severity is high (1): 40% model, 60% severity
    This way, scary incidents get pushed up, but mild ones stay model-driven.
    """
    severity_weight = 0.30 + 0.30 * severity_score
    model_weight = 1.0 - severity_weight
    blended = model_weight * model_proba + severity_weight * severity_score
    return min(1.0, blended)


def get_metrics(y_true, y_pred):
    """
    Computes the four main classification metrics.
    Returns them as a dictionary so we can build comparison tables.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# SECTION 6: DATA LOADING
# Loads the CSV file and caches it so it doesn't reload on every click.

@st.cache_data(show_spinner=False)
def load_data():
    """Load the training dataset from CSV."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if "label_recalled" not in df.columns:
        st.error("Missing target column: label_recalled")
        st.stop()
    return df


# SECTION 7: PREPROCESSOR BUILDER
# This builds the ColumnTransformer that turns raw text + categorical
# columns into numeric features the models can use.

def build_preprocessor(numeric_cols):
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


# SECTION 8: MODEL TRAINING
# This is the main function that trains all 9 models and returns
# everything the app needs: metrics, predictions, confusion matrix,
# keyword analysis, etc.
#
# It's cached with @st.cache_resource so it only runs once,
# even if the user switches pages.

@st.cache_resource(show_spinner=False)
def train_all_models(_df):
    """
    Trains the full model suite on the dataset:
      1. Logistic Regression (grid-searched over C and class_weight)
      2. Linear SVM (grid-searched over C)
      3. ComplementNB (grid-searched over alpha)
      4. XGBoost (gradient boosted trees)
      5. Random Forest
      6. SGD Classifier (stochastic gradient descent with log loss)
      7. Logistic Regression + SMOTE oversampling
      8. Linear SVM + SMOTE oversampling
      9. Logistic Regression with threshold tuning

    Also runs category-specific keyword analysis.
    """

    # Step 1: Prepare the data

    # Keep only the columns we need
    model_df = _df[REQUIRED_COLS].copy()

    # Fill missing values with "unknown" for all text/categorical columns
    for col in REQUIRED_COLS:
        if col != "label_recalled":
            model_df[col] = model_df[col].fillna("unknown").astype(str)

    # Step 2: Engineer numeric features

    # Count words in product description
    model_df["product_word_count"] = model_df["Product Description"].apply(
        lambda x: len(clean_text(x).split())
    )

    # Count words in incident description
    model_df["incident_word_count"] = model_df["Incident Description"].apply(
        lambda x: len(clean_text(x).split())
    )

    # Count hazard words in product description
    model_df["product_hazard_count"] = model_df["Product Description"].apply(
        lambda x: keyword_count(x, HAZARD_WORDS)
    )

    # Count hazard words in incident description
    model_df["incident_hazard_count"] = model_df["Incident Description"].apply(
        lambda x: keyword_count(x, HAZARD_WORDS)
    )

    # Binary flag: does the product have a model number?
    model_df["has_model_info"] = (
        model_df["Model Name or Number"].str.lower() != "unknown"
    ).astype(int)

    # List of our numeric feature columns
    numeric_cols = [
        "product_word_count",
        "incident_word_count",
        "product_hazard_count",
        "incident_hazard_count",
        "has_model_info",
    ]

    # Step 3: Train/test split (80/20, stratified)

    X = model_df.drop(columns=["label_recalled"])
    y = model_df["label_recalled"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Build the preprocessor
    preprocessor = build_preprocessor(numeric_cols)

    # 5-fold stratified cross-validation (used for grid search and threshold tuning)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Dictionary to store results from all models
    all_results = {}

    # Model 1: Logistic Regression (grid search over C and class_weight)

    lr_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=4000, solver="liblinear")),
    ])

    lr_param_grid = {
        "clf__C": [0.5, 1.0, 2.0],
        "clf__class_weight": [
            {0: 1, 1: 1},
            {0: 1, 1: 2},
            {0: 1, 1: 3},
            {0: 1, 1: 4},
        ],
    }

    lr_grid = GridSearchCV(
        lr_pipe, lr_param_grid,
        scoring="f1", cv=cv, n_jobs=-1, verbose=0,
    )
    lr_grid.fit(X_train, y_train)

    # Keep the best LR model and refit on full training set
    best_lr = lr_grid.best_estimator_
    best_lr.fit(X_train, y_train)

    # Get predictions and probabilities on test set
    lr_pred = best_lr.predict(X_test)
    lr_proba = best_lr.predict_proba(X_test)[:, 1]

    all_results["Logistic Regression"] = get_metrics(y_test, lr_pred)

    # Model 2: Linear SVM (grid search over C)

    svm_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LinearSVC(class_weight="balanced", max_iter=8000)),
    ])

    svm_param_grid = {"clf__C": [0.25, 0.5, 1.0, 2.0]}

    svm_grid = GridSearchCV(
        svm_pipe, svm_param_grid,
        scoring="f1", cv=cv, n_jobs=-1, verbose=0,
    )
    svm_grid.fit(X_train, y_train)

    best_svm = svm_grid.best_estimator_
    best_svm.fit(X_train, y_train)
    svm_pred = best_svm.predict(X_test)

    all_results["Linear SVM"] = get_metrics(y_test, svm_pred)

    # Model 3: ComplementNB (grid search over alpha)

    cnb_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", ComplementNB()),
    ])

    cnb_param_grid = {"clf__alpha": [0.5, 1.0, 2.0]}

    cnb_grid = GridSearchCV(
        cnb_pipe, cnb_param_grid,
        scoring="f1", cv=cv, n_jobs=-1, verbose=0,
    )
    cnb_grid.fit(X_train, y_train)

    best_cnb = cnb_grid.best_estimator_
    best_cnb.fit(X_train, y_train)
    cnb_pred = best_cnb.predict(X_test)

    all_results["ComplementNB"] = get_metrics(y_test, cnb_pred)

    # Transform data to dense arrays for models that need them
    # (XGBoost, Random Forest, SGD, and SMOTE all need dense input)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Convert sparse matrices to dense arrays
    if hasattr(X_train_transformed, "toarray"):
        X_train_dense = X_train_transformed.toarray()
    else:
        X_train_dense = X_train_transformed

    if hasattr(X_test_transformed, "toarray"):
        X_test_dense = X_test_transformed.toarray()
    else:
        X_test_dense = X_test_transformed

    # Calculate class imbalance ratio for XGBoost
    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    scale_pos = neg_count / max(pos_count, 1)

    # Model 4: XGBoost
    # Gradient boosted trees with scale_pos_weight to handle imbalance

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
    )
    xgb_model.fit(X_train_dense, y_train)
    xgb_pred = xgb_model.predict(X_test_dense)

    all_results["XGBoost"] = get_metrics(y_test, xgb_pred)

    # Model 5: Random Forest
    # Bagged trees with balanced class weights

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_dense, y_train)
    rf_pred = rf_model.predict(X_test_dense)

    all_results["Random Forest"] = get_metrics(y_test, rf_pred)

    # Model 6: SGD Classifier
    # Stochastic gradient descent with log loss (basically fast LR)

    sgd_model = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    sgd_model.fit(X_train_dense, y_train)
    sgd_pred = sgd_model.predict(X_test_dense)

    all_results["SGD Classifier"] = get_metrics(y_test, sgd_pred)

    # ------------------------------------------------------------------
    # Models 7 & 8: SMOTE oversampling + LR and SVM
    # SMOTE creates synthetic examples of the minority class (recalled)
    # to balance the training data before fitting the model.
    # Only runs if imbalanced-learn is installed.
    # ------------------------------------------------------------------

    smote_lr_pred = None
    smote_svm_pred = None

    if HAS_SMOTE:
        # Apply SMOTE to the training data only
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train_dense, y_train)

        # Model 7: LR on SMOTE-balanced data
        smote_lr = LogisticRegression(max_iter=4000, C=1.0, solver="liblinear")
        smote_lr.fit(X_smote, y_smote)
        smote_lr_pred = smote_lr.predict(X_test_dense)
        all_results["LR + SMOTE"] = get_metrics(y_test, smote_lr_pred)

        # Model 8: Linear SVM on SMOTE-balanced data
        smote_svm = LinearSVC(max_iter=8000)
        smote_svm.fit(X_smote, y_smote)
        smote_svm_pred = smote_svm.predict(X_test_dense)
        all_results["Linear SVM + SMOTE"] = get_metrics(y_test, smote_svm_pred)

    # Model 9: Threshold tuning for Logistic Regression
    # Instead of using the default 0.50 cutoff, we find the threshold
    # that maximizes F1 using out-of-fold cross-validation predictions.

    # Get out-of-fold probabilities from the training set
    oof_proba = cross_val_predict(
        best_lr, X_train, y_train,
        cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]

    # Try thresholds from 0.10 to 0.90 and pick the one with best F1
    thresholds = np.arange(0.10, 0.91, 0.05)
    best_threshold = 0.5
    best_threshold_f1 = 0.0

    for t in thresholds:
        preds_at_t = (oof_proba >= t).astype(int)
        f1_at_t = f1_score(y_train, preds_at_t, zero_division=0)
        if f1_at_t > best_threshold_f1:
            best_threshold_f1 = f1_at_t
            best_threshold = t

    # Apply the best threshold to the test set
    tuned_pred = (lr_proba >= best_threshold).astype(int)
    tuned_name = f"Best LR - Tuned {best_threshold:.2f}"
    all_results[tuned_name] = get_metrics(y_test, tuned_pred)

    # Find the best model overall (by F1 score)

    best_model_name = max(all_results, key=lambda k: all_results[k]["f1"])
    best_model_metrics = all_results[best_model_name]

    # Build a map of model name -> predictions for confusion matrix
    pred_map = {
        "Logistic Regression": lr_pred,
        "Linear SVM": svm_pred,
        "ComplementNB": cnb_pred,
        "XGBoost": xgb_pred,
        "Random Forest": rf_pred,
        "SGD Classifier": sgd_pred,
        tuned_name: tuned_pred,
    }
    if smote_lr_pred is not None:
        pred_map["LR + SMOTE"] = smote_lr_pred
    if smote_svm_pred is not None:
        pred_map["Linear SVM + SMOTE"] = smote_svm_pred

    # Get confusion matrix for the best model
    best_preds = pred_map.get(best_model_name, lr_pred)
    best_cm = confusion_matrix(y_test, best_preds)

    # Category-specific keyword analysis
    # For each product category with enough data, we train a small
    # logistic regression on just that category's text and extract
    # the top positive/negative coefficients as "keywords."

    category_counts = _df["Product Category"].value_counts()
    top_categories = category_counts[category_counts >= 50].index.tolist()

    cat_results = {}
    cat_keywords = {}

    for cat in top_categories:
        # Check we have enough data in train and test for this category
        cat_train_mask = X_train["Product Category"] == cat
        cat_test_mask = X_test["Product Category"] == cat
        n_train = cat_train_mask.sum()
        n_test = cat_test_mask.sum()

        if n_train < 20 or n_test < 5:
            continue

        y_cat_test = y_test[cat_test_mask]
        if y_cat_test.nunique() < 2:
            continue

        # Get metrics for this category using the best LR model
        cat_pred = best_lr.predict(X_test[cat_test_mask])
        cat_results[cat] = get_metrics(y_cat_test, cat_pred)
        cat_results[cat]["rows"] = int(n_train + n_test)
        cat_results[cat]["positive_rate"] = float(
            _df[_df["Product Category"] == cat]["label_recalled"].mean()
        )

        # Train a small LR on this category's text to get keywords
        cat_subset = model_df[model_df["Product Category"] == cat].copy()
        if cat_subset["label_recalled"].nunique() < 2 or len(cat_subset) < 30:
            continue

        try:
            # Simple TF-IDF on combined text for this category
            cat_tfidf = TfidfVectorizer(
                stop_words=list(ALL_STOPWORDS),
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                max_features=3000,
                sublinear_tf=True,
            )
            combined_text = (
                cat_subset["Product Description"] + " "
                + cat_subset["Incident Description"]
            )
            X_cat = cat_tfidf.fit_transform(combined_text)
            y_cat = cat_subset["label_recalled"]

            # Train a simple LR with higher weight on recalled class
            cat_lr = LogisticRegression(
                max_iter=3000,
                solver="liblinear",
                class_weight={0: 1, 1: 3},
                C=1.0,
            )
            cat_lr.fit(X_cat, y_cat)

            # Extract top positive and negative coefficients
            coefs = cat_lr.coef_[0]
            feature_names = cat_tfidf.get_feature_names_out()

            # Top 15 words pushing toward recalled
            top_pos_idx = np.argsort(coefs)[-15:][::-1]
            # Top 15 words pushing toward not recalled
            top_neg_idx = np.argsort(coefs)[:15]

            cat_keywords[cat] = {
                "positive": [
                    (feature_names[i], float(coefs[i])) for i in top_pos_idx
                ],
                "negative": [
                    (feature_names[i], float(coefs[i])) for i in top_neg_idx
                ],
            }
        except Exception:
            # Skip this category if anything goes wrong
            pass

    # Return everything the app pages need

    return {
        "best_lr": best_lr,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "all_results": all_results,
        "best_name": best_model_name,
        "best_metrics": best_model_metrics,
        "best_cm": best_cm,
        "best_threshold": best_threshold,
        "cat_results": cat_results,
        "cat_keywords": cat_keywords,
        "top_categories": top_categories,
    }


# SECTION 9: LOAD DATA AND TRAIN MODELS
# This runs when the app first starts. It's cached, so it only
# trains once even if the user navigates between pages.

with st.spinner("Loading data and training models... this may take 1-2 minutes on first load..."):
    df = load_data()
    bundle = train_all_models(df)


# SECTION 10: SIDEBAR NAVIGATION
st.sidebar.markdown("## RecallGuard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Dataset Insights", "Category Insights", "Risk Scorer", "Model Details"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Capstone Project  ·  Youssef Rafik\n\n"
    "Southern Connecticut State University\n\n"
    "Data Science  ·  Spring 2026"
)

if page == "Dataset Insights":

    st.markdown("## Dataset Insights")

    # Summary stats
    total = len(df)
    recalled = int(df["label_recalled"].sum())
    not_recalled = total - recalled
    st.caption(f"{total:,} incident reports  |  "
               f"{recalled:,} recalled  |  "
               f"{not_recalled:,} not recalled")

    col1, col2 = st.columns(2)

    # Left column: class balance bar chart
    with col1:
        st.markdown(
            '<div class="section-header">Class Balance</div>',
            unsafe_allow_html=True,
        )

        counts = df["label_recalled"].value_counts().sort_index()
        fig_balance = go.Figure(go.Bar(
            x=["Not Recalled (0)", "Recalled (1)"],
            y=counts.values,
            marker_color=["#334155", "#3b82f6"],
            text=counts.values,
            textposition="outside",
            textfont=dict(family="Space Mono", size=14),
        ))
        fig_balance.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=True, gridcolor="#1e293b"),
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig_balance, use_container_width=True)

    # Right column: recall rate by category
    with col2:
        st.markdown(
            '<div class="section-header">Top Categories by Recall Rate</div>',
            unsafe_allow_html=True,
        )

        # Group by category and calculate mean recall rate
        cat_stats = df.groupby("Product Category")["label_recalled"].agg(
            ["mean", "count"]
        )
        # Only show categories with at least 20 rows
        cat_stats = cat_stats[cat_stats["count"] >= 20]
        cat_stats = cat_stats.sort_values("mean", ascending=True)

        fig_categories = go.Figure(go.Bar(
            y=cat_stats.index,
            x=cat_stats["mean"],
            orientation="h",
            marker_color="#3b82f6",
            text=[f"{v:.1%}" for v in cat_stats["mean"]],
            textposition="outside",
            textfont=dict(family="Space Mono", size=11),
        ))
        fig_categories.update_layout(
            height=max(320, len(cat_stats) * 28),
            margin=dict(l=20, r=50, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#1e293b", tickformat=".0%"),
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig_categories, use_container_width=True)

    # Expandable data preview
    with st.expander("Preview data (first 50 rows)"):
        preview_cols = [
            "Product Description", "Product Category",
            "Brand", "Incident Description", "label_recalled",
        ]
        existing_cols = [c for c in preview_cols if c in df.columns]
        st.dataframe(
            df[existing_cols].head(50),
            use_container_width=True,
            height=350,
        )


# PAGE 3: RISK SCORER
# The user enters product and incident details, and the app returns
# a composite risk score that blends:
#   1. The ML model's probability (from historical recall patterns)
#   2. A hazard severity signal (from how dangerous the text sounds)

elif page == "Category Insights":

    st.markdown("## Category Insights")
    st.caption(
        "Explore recall patterns and key risk terms within each "
        "product category."
    )

    # Get the categories that have keyword analysis available
    available_cats = sorted(bundle["cat_keywords"].keys())

    if not available_cats:
        st.info("Not enough data for category-specific keyword analysis.")
        st.stop()

    # Dropdown to pick a category
    selected_category = st.selectbox(
        "Select a category", available_cats, index=0
    )

    if selected_category:

        # Get this category's results and keywords
        cr = bundle["cat_results"].get(selected_category, {})
        kw = bundle["cat_keywords"].get(selected_category, {})

        # Show 4 metric cards for this category
        st.markdown("")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Rows", f"{cr.get('rows', 0):,}")
        mc2.metric("Recall Rate", f"{cr.get('positive_rate', 0):.1%}")
        mc3.metric("Model F1", f"{cr.get('f1', 0):.3f}")
        mc4.metric("Model Recall", f"{cr.get('recall', 0):.3f}")

        st.markdown("")

        # Two columns: positive keywords on left, negative on right
        col_pos, col_neg = st.columns(2)

        # Left column: keywords that push TOWARD recall
        with col_pos:
            st.markdown(
                '<div class="section-header">'
                'Keywords — Higher Recall Risk</div>',
                unsafe_allow_html=True,
            )

            if kw.get("positive"):
                # Show keywords as colored chips
                chips_html = ""
                for word, coef in kw["positive"]:
                    chips_html += (
                        f'<span class="kw-chip kw-pos">'
                        f'{word} ({coef:.2f})</span>'
                    )
                st.markdown(chips_html, unsafe_allow_html=True)

                # Also show as a bar chart
                pos_df = pd.DataFrame(
                    kw["positive"], columns=["keyword", "coefficient"]
                )
                fig_pos = go.Figure(go.Bar(
                    x=pos_df["coefficient"],
                    y=pos_df["keyword"],
                    orientation="h",
                    marker_color="#ef4444",
                ))
                fig_pos.update_layout(
                    height=max(280, len(pos_df) * 22),
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(autorange="reversed"),
                    xaxis=dict(showgrid=True, gridcolor="#1e293b"),
                    font=dict(family="DM Sans", size=11),
                )
                st.plotly_chart(fig_pos, use_container_width=True)
            else:
                st.info("Not enough data for keyword analysis.")

        # Right column: keywords that push AWAY from recall
        with col_neg:
            st.markdown(
                '<div class="section-header">'
                'Keywords — Lower Recall Risk</div>',
                unsafe_allow_html=True,
            )

            if kw.get("negative"):
                # Show keywords as colored chips
                chips_html = ""
                for word, coef in kw["negative"]:
                    chips_html += (
                        f'<span class="kw-chip kw-neg">'
                        f'{word} ({coef:.2f})</span>'
                    )
                st.markdown(chips_html, unsafe_allow_html=True)

                # Also show as a bar chart
                neg_df = pd.DataFrame(
                    kw["negative"], columns=["keyword", "coefficient"]
                )
                fig_neg = go.Figure(go.Bar(
                    x=neg_df["coefficient"],
                    y=neg_df["keyword"],
                    orientation="h",
                    marker_color="#10b981",
                ))
                fig_neg.update_layout(
                    height=max(280, len(neg_df) * 22),
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(autorange="reversed"),
                    xaxis=dict(showgrid=True, gridcolor="#1e293b"),
                    font=dict(family="DM Sans", size=11),
                )
                st.plotly_chart(fig_neg, use_container_width=True)
            else:
                st.info("Not enough data for keyword analysis.")

        # Expandable explanation of coefficients
        with st.expander("How to read the keyword coefficients"):
            st.markdown("""
Coefficients come from a Logistic Regression model trained only on
text within this category. **Positive values** mean the keyword
pushes the model toward predicting a recall. **Negative values**
push toward *not recalled*.

These are not standalone risk scores, they reflect the learned
association within the full TF-IDF feature space for this category.
            """)
elif page == "Risk Scorer":

    st.markdown("## Risk Scorer")
    st.caption(
        "Enter product and incident details to estimate recall risk. "
        "Combines the ML model prediction with a hazard-severity "
        "analysis for a practical composite score."
    )

    # --- Professor-requested feature: load examples from the dataset ---
    # Initialize session_state keys for pre-filling the form fields.
    # These get set when the user clicks "Load Random Example" or picks
    # a record, and cleared when they click "Clear Fields."
    field_keys = [
        "ex_product_desc", "ex_incident_desc", "ex_brand",
        "ex_manufacturer", "ex_model_num", "ex_category",
        "ex_subcategory", "ex_type",
    ]
    for fk in field_keys:
        if fk not in st.session_state:
            st.session_state[fk] = ""

    # --- Professor-requested feature: browse and load examples from dataset ---
    # Step 1: pick a category, Step 2: see product titles, Step 3: select one
    # This auto-fills the form fields below so the user doesn't have to copy/paste.

    browse_categories = sorted(df["Product Category"].dropna().unique().tolist())

    br1, br2 = st.columns([1, 2])

    with br1:
        browse_cat = st.selectbox(
            "Browse by category",
            [""] + browse_categories,
            key="browse_cat",
        )

    with br2:
        if browse_cat:
            # Filter products in this category and build a readable label
            cat_rows = df[df["Product Category"] == browse_cat].copy()
            cat_rows["_label"] = (
                cat_rows["Product Description"].fillna("").astype(str).str[:80]
                + "  |  "
                + cat_rows["Brand"].fillna("").astype(str)
            )
            product_options = cat_rows["_label"].tolist()
            selected_product = st.selectbox(
                f"Select a product ({len(cat_rows):,} records)",
                [""] + product_options,
                key="browse_product",
            )
        else:
            selected_product = ""
            st.selectbox("Select a product", ["Pick a category first"], disabled=True)

    # Buttons: load selected product or clear fields
    load_col, random_col, clear_col = st.columns(3)

    with load_col:
        load_clicked = st.button("Load Selected Product", use_container_width=True)

    with random_col:
        random_clicked = st.button("Load Random Example", use_container_width=True)

    with clear_col:
        clear_clicked = st.button("Clear Fields", use_container_width=True)

    # Helper to fill session_state from a DataFrame row
    def _fill_from_row(sample):
        st.session_state["ex_product_desc"] = str(sample.get("Product Description", "") or "")
        st.session_state["ex_incident_desc"] = str(sample.get("Incident Description", "") or "")
        st.session_state["ex_brand"] = str(sample.get("Brand", "") or "")
        st.session_state["ex_manufacturer"] = str(sample.get("Manufacturer / Importer / Private Labeler Name", "") or "")
        st.session_state["ex_model_num"] = str(sample.get("Model Name or Number", "") or "")
        st.session_state["ex_category"] = str(sample.get("Product Category", "") or "")
        st.session_state["ex_subcategory"] = str(sample.get("Product Sub Category", "") or "")
        st.session_state["ex_type"] = str(sample.get("Product Type", "") or "")

    if load_clicked and browse_cat and selected_product:
        # Find the row matching the selected label
        cat_rows = df[df["Product Category"] == browse_cat].copy()
        cat_rows["_label"] = (
            cat_rows["Product Description"].fillna("").astype(str).str[:80]
            + "  |  "
            + cat_rows["Brand"].fillna("").astype(str)
        )
        match = cat_rows[cat_rows["_label"] == selected_product]
        if len(match) > 0:
            _fill_from_row(match.iloc[0])
            st.rerun()

    if random_clicked:
        _fill_from_row(df.sample(1).iloc[0])
        st.rerun()

    if clear_clicked:
        for fk in field_keys:
            st.session_state[fk] = ""
        st.rerun()

    st.markdown("---")

    # Get the unique values for the dropdown menus
    categories = sorted(df["Product Category"].dropna().unique().tolist())
    subcategories = sorted(df["Product Sub Category"].dropna().unique().tolist())
    product_types = sorted(df["Product Type"].dropna().unique().tolist())

    # Helper to find the index of a value in a list, or 0 if not found
    def _idx(lst, val):
        """Find index of val in [''] + lst, return 0 if not found."""
        full = [""] + lst
        try:
            return full.index(val)
        except ValueError:
            return 0

    # Build the input form — fields are pre-filled from session_state
    # but remain fully editable so the user can modify them.
    with st.form("risk_form"):

        # Row 1: text areas for descriptions
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            product_desc = st.text_area(
                "Product Description",
                value=st.session_state["ex_product_desc"],
                placeholder="e.g. Samsung top load washing machine, white",
                height=100,
            )
        with r1c2:
            incident_desc = st.text_area(
                "Incident / Complaint Text",
                value=st.session_state["ex_incident_desc"],
                placeholder="e.g. The washing machine exploded during spin cycle...",
                height=100,
            )

        # Row 2: brand, manufacturer, model
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            brand = st.text_input(
                "Brand",
                value=st.session_state["ex_brand"],
                placeholder="e.g. Samsung",
            )
        with r2c2:
            manufacturer = st.text_input(
                "Manufacturer / Importer / Private Labeler",
                value=st.session_state["ex_manufacturer"],
                placeholder="e.g. Samsung Electronics America",
            )
        with r2c3:
            model_num = st.text_input(
                "Model Name or Number",
                value=st.session_state["ex_model_num"],
                placeholder="e.g. WA50F9A8DSW",
            )

        # Row 3: category dropdowns (pre-selected from session_state)
        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            selected_category = st.selectbox(
                "Product Category",
                [""] + categories,
                index=_idx(categories, st.session_state["ex_category"]),
            )
        with r3c2:
            selected_subcategory = st.selectbox(
                "Product Sub Category",
                [""] + subcategories,
                index=_idx(subcategories, st.session_state["ex_subcategory"]),
            )
        with r3c3:
            selected_type = st.selectbox(
                "Product Type",
                [""] + product_types,
                index=_idx(product_types, st.session_state["ex_type"]),
            )

        # Submit button
        submitted = st.form_submit_button(
            "Estimate Recall Risk",
            use_container_width=True,
            type="primary",
        )

    # When the form is submitted, compute and display the risk score
    if submitted:

        # Make sure the user entered at least some text
        if not product_desc.strip() and not incident_desc.strip():
            st.warning("Please enter at least a product description or incident text.")
        else:

            # Build a single-row dataFrame matching the pipeline's columns
            input_row = pd.DataFrame([{
                "Product Description": product_desc or "unknown",
                "Incident Description": incident_desc or "unknown",
                "Manufacturer / Importer / Private Labeler Name": manufacturer or "unknown",
                "Brand": brand or "unknown",
                "Model Name or Number": model_num or "unknown",
                "Product Category": selected_category or "unknown",
                "Product Sub Category": selected_subcategory or "unknown",
                "Product Type": selected_type or "unknown",
            }])

            # Add the engineered numeric features
            input_row["product_word_count"] = len(clean_text(product_desc).split())
            input_row["incident_word_count"] = len(clean_text(incident_desc).split())
            input_row["product_hazard_count"] = keyword_count(product_desc, HAZARD_WORDS)
            input_row["incident_hazard_count"] = keyword_count(incident_desc, HAZARD_WORDS)
            input_row["has_model_info"] = int(
                (model_num or "").strip().lower() not in ("", "unknown")
            )

            # Get the ML model's probability
            best_lr = bundle["best_lr"]
            try:
                model_proba = best_lr.predict_proba(input_row)[:, 1][0]
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

            # Compute the hazard severity signal from the text
            full_text = (product_desc or "") + " " + (incident_desc or "")
            sev_score, crit_hits, high_hits, mod_hits = compute_severity_score(full_text)

            # Blend model probability and severity into a composite score
            final_score = composite_risk_score(model_proba, sev_score)

            # Classify into Low / Moderate / High based on composite score
            if final_score < 0.25:
                bucket = "Low"
                css_class = "risk-low"
                action = "No immediate action needed. Monitor for new reports."
            elif final_score < 0.55:
                bucket = "Moderate"
                css_class = "risk-moderate"
                action = (
                    "Review product reports. Consider proactive quality "
                    "checks and gather more incident data."
                )
            else:
                bucket = "High"
                css_class = "risk-high"
                action = (
                    "Escalate for detailed safety review. Gather related "
                    "incident data and consider contacting the manufacturer."
                )

            # Display the big composite score box
            st.markdown("")
            rc1, rc2, rc3 = st.columns([1, 2, 1])
            with rc2:
                st.markdown(
                    f'<div class="risk-box {css_class}">'
                    f'<h2>{final_score:.0%}</h2>'
                    f'<p>Composite Risk Score &middot; '
                    f'<strong>{bucket}</strong></p></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")

            # Show the two individual signals side by side
            sig1, sig2 = st.columns(2)
            with sig1:
                st.metric(
                    "ML Model Probability",
                    f"{model_proba:.1%}",
                    help=(
                        "Probability from the trained Logistic Regression "
                        "pipeline based on historical CPSC recall patterns"
                    ),
                )
            with sig2:
                st.metric(
                    "Hazard Severity Signal",
                    f"{sev_score:.0%}",
                    help=(
                        "Rule-based score from the severity of language "
                        "in the incident description (critical, high, "
                        "moderate hazard terms)"
                    ),
                )

            st.markdown("")
            st.markdown(f"**Suggested Action:** {action}")

            # Expandable breakdown of what influenced the score
            with st.expander("What influenced this score?"):

                # Show the math behind the composite score
                st.markdown("**Composite Score Breakdown**")
                sev_weight = 0.30 + 0.30 * sev_score
                model_weight = 1.0 - sev_weight
                st.markdown(
                    f"- ML model contribution: {model_proba:.1%} "
                    f"x {model_weight:.0%} weight = "
                    f"{model_proba * model_weight:.1%}"
                )
                st.markdown(
                    f"- Severity contribution: {sev_score:.0%} "
                    f"x {sev_weight:.0%} weight = "
                    f"{sev_score * sev_weight:.1%}"
                )

                # Show which hazard words were detected
                st.markdown("")
                st.markdown("**Hazard Language Detected**")

                if crit_hits > 0:
                    crit_found = [
                        w for w in SEVERITY_CRITICAL
                        if w in clean_text(full_text)
                    ]
                    st.markdown(
                        f"- Critical terms ({crit_hits}): "
                        + ", ".join(sorted(crit_found)[:6])
                    )
                if high_hits > 0:
                    high_found = [
                        w for w in SEVERITY_HIGH
                        if w in clean_text(full_text)
                    ]
                    st.markdown(
                        f"- High-severity terms ({high_hits}): "
                        + ", ".join(sorted(high_found)[:6])
                    )
                if mod_hits > 0:
                    mod_found = [
                        w for w in SEVERITY_MODERATE
                        if w in clean_text(full_text)
                    ]
                    st.markdown(
                        f"- Moderate terms ({mod_hits}): "
                        + ", ".join(sorted(mod_found)[:6])
                    )
                if crit_hits == 0 and high_hits == 0 and mod_hits == 0:
                    st.markdown("- No hazard-related language detected")

                # Show other factors
                st.markdown("")
                st.markdown("**Other Factors**")
                if selected_category and selected_category in bundle["cat_results"]:
                    cr = bundle["cat_results"][selected_category]
                    st.markdown(
                        f"- Category *{selected_category}* has a base "
                        f"recall rate of {cr['positive_rate']:.1%}"
                    )
                if model_num and model_num.strip().lower() not in ("", "unknown"):
                    st.markdown(
                        "- Model number provided; may match known recalled products"
                    )
                else:
                    st.markdown(
                        "- No model number provided; less matching signal available"
                    )

            # Expandable explanation of why we use two signals
            with st.expander("Why two signals?"):
                st.markdown("""
The ML model was trained to predict **formal CPSC recalls**, not whether
a product is dangerous. Most dangerous incidents never lead to a formal
recall because recall decisions depend on factors the model cannot see
(legal negotiations, complaint volume, manufacturer cooperation).

This means a genuinely alarming incident (like a battery explosion
causing injury) can get a low ML probability if products in that
category were rarely recalled in the training data.

To make the tool practically useful, we combine two signals:

- **ML Model Probability** : learned patterns from historical recall
  data (brand, model, category, text features)
- **Hazard Severity Signal** : rule-based analysis of how dangerous
  the incident language sounds (critical terms like "death",
  "explosion", "hospitalized" score higher than moderate terms like
  "defective" or "pain")

The composite score gives more weight to the severity signal when the
incident language is alarming.
                """)


# PAGE 4: CATEGORY INSIGHTS
# For each product category, shows:
# - Row count and recall rate
# - Model performance within that category
# - Top keywords pushing toward/away from recall


# PAGE: MODEL DETAILS (was Dashboard)
# Shows the best model's metrics, confusion matrix, and a comparison
# table and bar chart of all 9 models ranked by F1 score.

elif page == "Model Details":

    # Hero banner at the top
    st.markdown("""
    <div class="hero">
        <h1>RecallGuard</h1>
        <p>A machine-learning system that estimates product recall risk
        from CPSC incident reports. Built with TF-IDF feature engineering,
        multi-model comparison, SMOTE oversampling, and threshold-tuned
        classification.</p>
    </div>
    """, unsafe_allow_html=True)

    # Get the best model's metrics
    m = bundle["best_metrics"]

    # Section label
    st.markdown(
        '<div class="section-header">Best Model Performance</div>',
        unsafe_allow_html=True,
    )

    # Display 5 metric cards in a row
    c1, c2, c3, c4, c5 = st.columns(5)

    display_name = bundle["best_name"]

    c1.metric("Best Model", display_name)
    c2.metric("Accuracy", f"{m['accuracy']:.4f}")
    c3.metric("Precision", f"{m['precision']:.4f}")
    c4.metric("Recall", f"{m['recall']:.4f}")
    c5.metric("F1 Score", f"{m['f1']:.4f}")

    st.markdown("")

    # Two columns: confusion matrix on left, comparison table on right
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown(
            '<div class="section-header">Confusion Matrix</div>',
            unsafe_allow_html=True,
        )

        # Build a Plotly heatmap for the confusion matrix
        cm = bundle["best_cm"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm[::-1],
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 1", "Actual 0"],
            text=cm[::-1],
            texttemplate="%{text}",
            textfont=dict(size=18, family="Space Mono"),
            colorscale=[[0, "#1e293b"], [1, "#3b82f6"]],
            showscale=False,
            hoverinfo="skip",
        ))
        fig_cm.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans"),
            xaxis=dict(side="bottom"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_right:
        st.markdown(
            '<div class="section-header">Full Model Comparison</div>',
            unsafe_allow_html=True,
        )

        # Build a DataFrame from all model results and sort by F1
        results_table = pd.DataFrame(bundle["all_results"]).T
        results_table.index.name = "Model"
        results_table = results_table.sort_values("f1", ascending=False)

        # Display as a styled table with the best values highlighted
        st.dataframe(
            results_table.style
                .format("{:.4f}")
                .highlight_max(axis=0, props="color: #3b82f6; font-weight: bold;"),
            use_container_width=True,
            height=380,
        )

    # F1 Score comparison bar chart
    st.markdown("")
    st.markdown(
        '<div class="section-header">F1 Score Comparison</div>',
        unsafe_allow_html=True,
    )

    bar_data = results_table.reset_index().rename(columns={"index": "Model"})

    # Highlight the best model in blue, rest in gray
    bar_colors = []
    for name in bar_data["Model"]:
        if name == bundle["best_name"]:
            bar_colors.append("#3b82f6")
        else:
            bar_colors.append("#334155")

    fig_f1 = go.Figure(go.Bar(
        x=bar_data["f1"],
        y=bar_data["Model"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.4f}" for v in bar_data["f1"]],
        textposition="outside",
        textfont=dict(family="Space Mono", size=11),
    ))
    fig_f1.update_layout(
        height=max(320, len(bar_data) * 38),
        margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor="#1e293b",
            range=[0, max(bar_data["f1"]) * 1.15],
        ),
        yaxis=dict(autorange="reversed"),
        font=dict(family="DM Sans", size=12),
    )
    st.plotly_chart(fig_f1, use_container_width=True)

    # Expandable explanation of the metrics
    with st.expander("How to interpret these metrics"):
        st.markdown("""
**Accuracy** : Fraction of all predictions that were correct. High here
because most products are *not* recalled (class imbalance).

**Precision** : Of the products the model flagged as recall-likely, how
many actually were recalled. Higher precision = fewer false alarms.

**Recall (Sensitivity)** : Of products that *were* recalled, how many
did the model correctly catch. Higher recall = fewer missed recalls.

**F1 Score** : Mean of precision and recall. The primary metric
for this project because both false alarms and missed recalls matter.

**Confusion Matrix** : Top-left is True Negatives, bottom-right is True
Positives. Off-diagonal cells are errors.

**SMOTE Models** : Synthetic Minority Over-sampling Technique generates
synthetic examples of the minority class (recalled) to address class
imbalance.

**Threshold Tuning** : Instead of using the default 0.50 cutoff for
Logistic Regression, the tuned version picks the threshold that
maximizes F1 on out-of-fold cross-validation predictions.

**XGBoost** : Gradient-boosted decision tree ensemble with
scale_pos_weight set to the class imbalance ratio.

**Random Forest** : Bagged decision tree ensemble with balanced
class weights.

**SGD Classifier** : Stochastic Gradient Descent with log loss
(equivalent to logistic regression optimized via SGD), using balanced
class weights.
        """)


# PAGE 2: DATASET INSIGHTS
# Shows class balance, recall rate by category, and a data preview.

