"""Fit, select, calibrate and evaluate sparse text classifiers."""

import logging

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.svm import LinearSVC

from recallguard.data import TARGET, feature_frame
from recallguard.evaluation import partitions, score_metrics

LOGGER = logging.getLogger(__name__)
FEATURE_SETS = {
    "all": {"word", "character", "category", "numeric"},
    "text": {"word", "character"},
    "word": {"word"},
}


def preprocessing(feature_set: str) -> ColumnTransformer:
    blocks = [
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=8000), "text"),
        (
            "character",
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=5000),
            "text",
        ),
        (
            "category",
            OneHotEncoder(handle_unknown="ignore", min_frequency=2),
            ["category", "brand", "manufacturer", "product_type"],
        ),
        ("numeric", MaxAbsScaler(), ["word_count"]),
    ]
    return ColumnTransformer(
        [block for block in blocks if block[0] in FEATURE_SETS[feature_set]],
        sparse_threshold=1.0,
    )


def classifiers(seed: int, ratio: float, smoke: bool):
    """Yield variants grouped by estimator and training-side imbalance strategy."""
    yield "dummy", "prior", DummyClassifier(strategy="prior")
    for weight in [None] if smoke else [None, "balanced"]:
        strategy = "unweighted" if weight is None else "weighted"
        for strength in [1.0] if smoke else [0.3, 1.0, 3.0]:
            yield (
                "logistic",
                f"{strategy}:{strength}",
                LogisticRegression(
                    C=strength,
                    class_weight=weight,
                    max_iter=2000,
                    solver="liblinear",
                    random_state=seed,
                ),
            )
            if not smoke:
                yield (
                    "linear_svm",
                    f"{strategy}:{strength}",
                    LinearSVC(C=strength, class_weight=weight, random_state=seed),
                )
    if not smoke:
        from xgboost import XGBClassifier

        for alpha in (0.3, 1.0):
            yield "complement_nb", f"alpha={alpha}", ComplementNB(alpha=alpha)
        for scale in (1.0, ratio):
            yield (
                "xgboost",
                f"positive_weight={scale:.3g}",
                XGBClassifier(
                    n_estimators=125,
                    max_depth=4,
                    n_jobs=2,
                    random_state=seed,
                    scale_pos_weight=scale,
                    eval_metric="logloss",
                ),
            )
        yield (
            "logistic",
            "random_oversampling",
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed),
        )


def raw_scores(pipeline: Pipeline, features) -> np.ndarray:
    if hasattr(pipeline, "decision_function"):
        return np.asarray(pipeline.decision_function(features), dtype=float)
    return np.asarray(pipeline.predict_proba(features)[:, 1], dtype=float)


def probability(calibrator, scores: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]


def threshold_for(target, scores) -> float:
    precision, recall, thresholds = precision_recall_curve(target, scores)
    combined = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    return float(thresholds[np.argmax(combined)])


def train(frame, design: str, seed: int, feature_set: str, smoke: bool = False) -> dict:
    """Freeze variant and threshold choices before computing held-out metrics."""
    (training, validation, calibration, test), groups = partitions(frame, design, seed)
    x_train, x_val, x_cal, x_test = map(
        feature_frame, (training, validation, calibration, test)
    )
    y_train, y_val, y_cal, y_test = (
        part[TARGET] for part in (training, validation, calibration, test)
    )
    preprocessor = preprocessing(feature_set).fit(x_train)
    transformed = preprocessor.transform(x_train)
    positives = np.flatnonzero(y_train.to_numpy() == 1)
    ratio = (len(y_train) - len(positives)) / len(positives)
    rng = np.random.default_rng(seed)
    oversample = np.concatenate(
        [
            np.arange(len(y_train)),
            rng.choice(
                positives, size=max(0, len(y_train) - 2 * len(positives)), replace=True
            ),
        ]
    )
    selected = {}
    variants = []
    for family, variant, classifier in classifiers(seed, ratio, smoke):
        LOGGER.info("Fitting %s / %s / %s", design, family, variant)
        if variant == "random_oversampling":
            classifier.fit(transformed[oversample], y_train.iloc[oversample])
        else:
            classifier.fit(transformed, y_train)
        pipeline = Pipeline([("preprocess", preprocessor), ("classifier", classifier)])
        val_raw = raw_scores(pipeline, x_val)
        ap = float(average_precision_score(y_val, val_raw))
        variants.append(
            {"model": family, "variant": variant, "validation_average_precision": ap}
        )
        if family not in selected or ap > selected[family][0]:
            selected[family] = (ap, variant, pipeline, val_raw)

    winner = max(
        selected, key=lambda family: selected[family][0] if family != "dummy" else -1.0
    )
    comparisons = {}
    artifact = {}
    for family, (ap, variant, pipeline, val_raw) in selected.items():
        calibration_raw = raw_scores(pipeline, x_cal)
        calibrator = LogisticRegression(C=10.0, solver="lbfgs").fit(
            calibration_raw.reshape(-1, 1), y_cal
        )
        # Reject direction reversal on the calibration split so priority ranks stay interpretable.
        if calibrator.coef_[0, 0] < 0:
            calibrator.fit(np.zeros_like(calibration_raw).reshape(-1, 1), y_cal)
        val_prob = probability(calibrator, val_raw)
        threshold = threshold_for(y_val, val_prob)
        low = min(threshold, float(np.quantile(val_prob, 0.75)))
        scores = probability(calibrator, raw_scores(pipeline, x_test))
        observed, predicted = calibration_curve(
            y_test, scores, n_bins=8, strategy="quantile"
        )
        report = score_metrics(y_test, scores, threshold)
        default = score_metrics(y_test, scores, 0.5)
        comparisons[family] = {
            **report,
            "variant": variant,
            "validation_average_precision": ap,
            "threshold": threshold,
            "default_threshold_f1": default["f1"],
            "brier": brier_score_loss(y_test, scores),
            "log_loss": log_loss(y_test, scores),
            "reliability": {
                "mean_score": predicted.tolist(),
                "observed_fraction": observed.tolist(),
            },
            "threshold_analysis": [
                {"threshold": float(t), "precision": float(p), "recall": float(r)}
                for p, r, t in zip(*precision_recall_curve(y_val, val_prob))
            ],
        }
        if family == winner:
            artifact = {
                "pipeline": pipeline,
                "calibrator": calibrator,
                "low": low,
                "high": threshold,
            }
    summaries = {}
    for name, part in zip(
        ("train", "validation", "calibration", "test"),
        (training, validation, calibration, test),
    ):
        dates = part.report_date.dropna()
        summaries[name] = {
            "rows": len(part),
            "positives": int(part[TARGET].sum()),
            "date_min": dates.min().isoformat() if len(dates) else None,
            "date_max": dates.max().isoformat() if len(dates) else None,
        }
    earlier = training.index.union(validation.index).union(calibration.index)
    comparisons[winner]["category_analysis"] = []
    y_test_array = y_test.to_numpy()
    winner_scores = probability(
        artifact["calibrator"], raw_scores(artifact["pipeline"], x_test)
    )
    for category, part in test.groupby("category"):
        if len(part) >= 20 and part[TARGET].nunique() == 2:
            mask = test.index.isin(part.index)
            comparisons[winner]["category_analysis"].append(
                {
                    "category": category or "Unspecified",
                    "rows": len(part),
                    "f1": score_metrics(
                        y_test_array[mask], winner_scores[mask], artifact["high"]
                    )["f1"],
                }
            )
    return {
        **artifact,
        "metadata": {
            "design": design,
            "seed": seed,
            "feature_set": feature_set,
            "smoke": smoke,
            "selected": winner,
            "partitions": summaries,
            "excluded_rows": len(frame)
            - sum(map(len, (training, validation, calibration, test))),
            "test_rows_sharing_earlier_product_proxy": int(
                groups.loc[test.index].isin(groups.loc[earlier]).sum()
            ),
            "test_rows_sharing_earlier_manufacturer": int(
                test.manufacturer.ne("")
                .mul(test.manufacturer.isin(frame.loc[earlier, "manufacturer"]))
                .sum()
            ),
            "variants": variants,
            "comparison": comparisons,
        },
    }
