"""Retrospective evaluation with model selection confined to development folds."""

import argparse
import json
from dataclasses import dataclass
from itertools import product

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from features import ALL_STOPWORDS, build_preprocessor, load_data, prepare_features
from overlap import overlap_counts, related_reports

SEED = 42
LR = "Logistic Regression"
TUNED_LR = "Logistic Regression (tuned threshold)"
THRESHOLDS = (0.5, *(i / 100 for i in range(10, 91, 5) if i != 50))


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: object
    smote: bool = False
    balanced_weights: bool = False

    def fit(self, X, y):
        kwargs = {"sample_weight": compute_sample_weight("balanced", y)} if self.balanced_weights else {}
        return clone(self.estimator).fit(X, y, **kwargs)


def candidates():
    options = [
        Candidate(LR, LogisticRegression(
            C=c, class_weight={0: 1, 1: weight}, solver="liblinear", max_iter=4000,
            random_state=SEED,
        ))
        for c, weight in product((0.5, 1.0, 2.0), (1, 2, 3, 4))
    ]
    options.extend(Candidate("Linear SVM", LinearSVC(
        C=c, class_weight="balanced", max_iter=8000, random_state=SEED,
    )) for c in (0.25, 0.5, 1.0, 2.0))
    options.extend(Candidate("ComplementNB", ComplementNB(alpha=alpha))
                   for alpha in (0.5, 1.0, 2.0))
    options.extend([
        Candidate("XGBoost", XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss",
            random_state=SEED, n_jobs=2,
        ), balanced_weights=True),
        Candidate("Random Forest", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=SEED, n_jobs=2,
        )),
        Candidate("SGD Classifier", SGDClassifier(
            loss="log_loss", alpha=1e-4, max_iter=2000, class_weight="balanced",
            random_state=SEED,
        )),
        Candidate("LR + SMOTE", LogisticRegression(
            max_iter=4000, solver="liblinear", random_state=SEED,
        ), smote=True),
        Candidate("Linear SVM + SMOTE", LinearSVC(
            max_iter=8000, random_state=SEED,
        ), smote=True),
    ])
    return options


def split_indices(y, groups=None):
    indices = np.arange(len(y))
    if groups is None:
        return train_test_split(indices, stratify=y, test_size=0.2, random_state=SEED)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    return next(splitter.split(indices, y, groups))


def validation_folds(y, groups=None):
    splitter_type = StratifiedKFold if groups is None else StratifiedGroupKFold
    splitter = splitter_type(n_splits=5, shuffle=True, random_state=SEED)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def metrics(y, predictions):
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
    }


def select_threshold(y, scores):
    return max(THRESHOLDS, key=lambda threshold: f1_score(
        y, scores >= threshold, zero_division=0,
    ))


@dataclass
class Selection:
    name: str
    candidate: Candidate
    threshold: float | None
    oof_predictions: np.ndarray
    cv_metrics: dict


def _resample(X, y):
    minority = int(np.bincount(np.asarray(y, dtype=int)).min())
    if minority < 2:
        raise ValueError("SMOTE requires at least two observations per class in each training fold.")
    return SMOTE(random_state=SEED, k_neighbors=min(5, minority - 1)).fit_resample(X, y)


def select_model(X, y, *, groups=None, options=None, preprocessor=None):
    """Compare configurations by pooled out-of-fold F1 on development rows only."""
    options = candidates() if options is None else options
    preprocessor = build_preprocessor() if preprocessor is None else preprocessor
    predictions = np.empty((len(options), len(y)), dtype=int)
    lr_scores = {i: np.empty(len(y)) for i, option in enumerate(options) if option.name == LR}
    for train, validation in validation_folds(y, groups):
        y_train = y.iloc[train]
        transformer = clone(preprocessor)
        train_features = transformer.fit_transform(X.iloc[train])
        validation_features = transformer.transform(X.iloc[validation])
        if any(option.smote for option in options):
            sampled = _resample(train_features, y_train)
        for i, option in enumerate(options):
            estimator = option.fit(*(sampled if option.smote else (train_features, y_train)))
            if i in lr_scores:
                scores = estimator.predict_proba(validation_features)[:, 1]
                lr_scores[i][validation] = scores
                predictions[i, validation] = scores > 0.5
            else:
                predictions[i, validation] = estimator.predict(validation_features)

    by_family = {}
    for i, option in enumerate(options):
        variants = [(option.name, None, predictions[i])]
        if i in lr_scores:
            threshold = select_threshold(y, lr_scores[i])
            variants.append((TUNED_LR, threshold, lr_scores[i] >= threshold))
        for name, threshold, labels in variants:
            result = Selection(name, option, threshold, labels, metrics(y, labels))
            if name not in by_family or result.cv_metrics["f1"] > by_family[name].cv_metrics["f1"]:
                by_family[name] = result
    winner = max(by_family.values(), key=lambda result: result.cv_metrics["f1"])
    return winner, by_family


def fit_selected(X, y, selection, preprocessor=None):
    transformer = build_preprocessor() if preprocessor is None else clone(preprocessor)
    features = transformer.fit_transform(X)
    args = _resample(features, y) if selection.candidate.smote else (features, y)
    estimator = selection.candidate.fit(*args)
    return Pipeline([("preprocessor", transformer), ("clf", estimator)])


def predict(model, X, threshold=None):
    if threshold is None:
        return model.predict(X)
    return (model.predict_proba(X)[:, 1] >= threshold).astype(int)


def develop(frame, groups=None):
    train, test = split_indices(frame["label_recalled"], groups)
    train_frame = frame.iloc[train]
    X, y = prepare_features(train_frame), train_frame["label_recalled"]
    winner, families = select_model(X, y, groups=None if groups is None else groups[train])
    model = fit_selected(X, y, winner)
    return {
        "train": train, "test": test, "X": X, "y": y,
        "winner": winner, "families": families, "model": model,
    }


def fit_scorer(frame):
    """Fit the scorer's logistic regression without running the full benchmark."""
    train, _ = split_indices(frame["label_recalled"])
    training = frame.iloc[train]
    X, y = prepare_features(training), training["label_recalled"]
    logistic_options = [option for option in candidates() if option.name == LR]
    _, families = select_model(X, y, options=logistic_options)
    return fit_selected(X, y, families[LR])


def evaluate_holdout(frame, development):
    """Score only the frozen winner and its frozen threshold on the holdout."""
    winner = development["winner"]
    test = frame.iloc[development["test"]]
    y = test["label_recalled"]
    predictions = predict(development["model"], prepare_features(test), winner.threshold)
    return {
        "rows": len(y), "positives": int(y.sum()),
        "metrics": metrics(y, predictions),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
    }


def category_insights(development):
    X, y, winner = (development[key] for key in ("X", "y", "winner"))
    results, keywords = {}, {}
    for category, count in X["Product Category"].value_counts().items():
        mask = X["Product Category"] == category
        y_cat = y[mask]
        if count < 50 or y_cat.nunique() < 2:
            continue
        results[category] = {
            **metrics(y_cat, winner.oof_predictions[mask.to_numpy()]),
            "rows": count, "positive_rate": float(y_cat.mean()),
        }
        vectorizer = TfidfVectorizer(
            stop_words=list(ALL_STOPWORDS), ngram_range=(1, 2), min_df=2,
            max_df=0.95, max_features=3000, sublinear_tf=True,
        )
        texts = X.loc[mask, "Product Description"] + " " + X.loc[mask, "Incident Description"]
        try:
            features = vectorizer.fit_transform(texts)
        except ValueError as error:
            if "no terms remain" not in str(error).lower() and "empty vocabulary" not in str(error).lower():
                raise
            continue
        estimator = LogisticRegression(
            solver="liblinear", class_weight={0: 1, 1: 3}, max_iter=3000, random_state=SEED,
        ).fit(features, y_cat)
        coefficients = zip(vectorizer.get_feature_names_out(), estimator.coef_[0])
        ordered = sorted(coefficients, key=lambda pair: pair[1])
        keywords[category] = {
            "positive": [(word, coef) for word, coef in ordered[::-1] if coef > 0][:15],
            "negative": [(word, coef) for word, coef in ordered if coef < 0][:15],
        }
    return results, keywords


def audit(frame, groups, buckets):
    y = frame["label_recalled"]
    output = {
        "columns": frame.columns.tolist(),
        "rows": len(frame), "exact_duplicate_records": int(frame.duplicated().sum()),
        "duplicate_report_numbers": int(frame["Report No."].dropna().duplicated().sum()),
        "related_components": int(len(set(groups))),
        "largest_component": int(np.unique(groups, return_counts=True)[1].max()),
        "match_coverage": {
            key: {"keys": len(values), "rows": len(set().union(*map(set, values.values())))}
            for key, values in buckets.items()
        },
    }
    for name, keys in (("random", None), ("grouped", groups)):
        train, test = split_indices(y, keys)
        output[name] = {
            "train_rows": len(train), "train_positives": int(y.iloc[train].sum()),
            "test_rows": len(test), "test_positives": int(y.iloc[test].sum()),
            "overlap": overlap_counts(buckets, train, test),
        }
    return output


def summarize(development, holdout):
    winner = development["winner"]
    return {
        "development_cv": {name: selection.cv_metrics
                           for name, selection in development["families"].items()},
        "selection": {"model": winner.name, "threshold": winner.threshold},
        "retrospective_holdout": holdout,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true", help="Inspect the split without fitting models")
    args = parser.parse_args()
    frame = load_data()
    groups, buckets = related_reports(frame)
    results = {"audit": audit(frame, groups, buckets)}
    if not args.audit_only:
        # Freeze both experiments before inspecting either holdout result.
        random = develop(frame)
        grouped = develop(frame, groups)
        results.update(random=summarize(random, evaluate_holdout(frame, random)),
                       grouped=summarize(grouped, evaluate_holdout(frame, grouped)))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
