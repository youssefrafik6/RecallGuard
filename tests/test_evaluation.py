import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

import evaluation
from evaluation import Candidate, LR, evaluate_holdout, select_threshold, split_indices, validation_folds
from features import FEATURE_COLS, build_preprocessor, prepare_features
from overlap import overlap_counts, related_reports


def reports(n=100):
    rows = []
    for i in range(n):
        positive = i % 2
        rows.append({
            **dict.fromkeys(FEATURE_COLS, "common"),
            "Product Description": "electric heater" if positive else "wooden table",
            "Incident Description": "severe smoke" if positive else "loose fixture",
            "Brand": f"brand{i}",
            "label_recalled": positive,
        })
    return pd.DataFrame(rows)


def test_preprocessor_transforms_unseen_text_and_categories():
    X = prepare_features(reports(20))
    transformer = build_preprocessor()
    fitted = transformer.fit_transform(X)
    unseen = X.iloc[[0]].assign(Brand="never-seen")
    unseen["Product Description"] = "unseenword"
    assert transformer.transform(unseen).shape == (1, fitted.shape[1])
    assert "unseenword" not in transformer.named_transformers_["product_word_tfidf"].vocabulary_
    assert "never-seen" not in transformer.named_transformers_["brand_ohe"].categories_[0]


def test_random_and_grouped_splits_keep_holdout_out_of_development():
    y = reports()["label_recalled"]
    grouped = np.repeat(np.arange(50), 2)
    for groups in [None, grouped]:
        train, test = split_indices(y, groups)
        assert not set(train) & set(test)
        if groups is not None:
            assert not set(groups[train]) & set(groups[test])
        seen = []
        for fit, validation in validation_folds(y.iloc[train], None if groups is None else groups[train]):
            assert not set(train[fit]) & set(train[validation])
            assert not set(test) & (set(train[fit]) | set(train[validation]))
            if groups is not None:
                assert not set(groups[train[fit]]) & set(groups[train[validation]])
            seen.extend(validation)
        assert sorted(seen) == list(range(len(train)))


def test_changing_only_holdout_labels_does_not_change_selection(monkeypatch):
    frame = reports()
    train, test = split_indices(frame["label_recalled"])
    # Lock the already-defined split. Stratifying again after changing labels
    # would change holdout membership and thereby change the development data.
    monkeypatch.setattr(evaluation, "split_indices", lambda y, groups: (train, test))
    monkeypatch.setattr(evaluation, "candidates", lambda: [
        Candidate("Always negative", DummyClassifier(strategy="constant", constant=0)),
        Candidate(LR, LogisticRegression(random_state=42)),
    ])
    original = evaluation.develop(frame)
    holdout = evaluate_holdout(frame, original)
    changed = frame.copy()
    changed.loc[test, "label_recalled"] = 1 - changed.loc[test, "label_recalled"]
    repeated = evaluation.develop(changed)
    assert original["winner"].name == repeated["winner"].name == LR
    assert original["winner"].threshold == repeated["winner"].threshold
    assert original["winner"].cv_metrics == repeated["winner"].cv_metrics
    assert holdout["metrics"]["f1"] > evaluate_holdout(changed, repeated)["metrics"]["f1"]
    inputs = prepare_features(frame.iloc[test])
    np.testing.assert_array_equal(original["model"].predict(inputs), repeated["model"].predict(inputs))


def test_threshold_selection_uses_development_scores():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    assert select_threshold(y, scores) == 0.25
    assert select_threshold(y, np.array([0.1, 0.2, 0.6, 0.7])) == 0.5


class TrackRows(TransformerMixin, BaseEstimator):
    fitted_rows = []

    def fit(self, X, y=None):
        self.fitted_rows.append(set(X.index))
        return self

    def transform(self, X):
        return X[["feature"]].to_numpy()


def test_smote_and_preprocessing_are_training_fold_local(monkeypatch):
    X = pd.DataFrame({"feature": np.arange(50)})
    y = pd.Series(np.arange(50) % 2)
    sampled_rows = []
    TrackRows.fitted_rows.clear()

    def resample(features, labels):
        sampled_rows.append(set(features[:, 0]))
        return features, labels

    monkeypatch.setattr(evaluation, "_resample", resample)
    winner, _ = evaluation.select_model(
        X, y, preprocessor=TrackRows(),
        options=[Candidate("SMOTE candidate", LogisticRegression(), smote=True)],
    )
    expected = [set(train) for train, _ in validation_folds(y)]
    assert sampled_rows == TrackRows.fitted_rows == expected
    assert set(winner.oof_predictions) <= {0, 1}


def test_grouping_joins_campaigns_models_and_incidents_but_not_manufacturer_alone():
    base = "the portable heater was running in the spare bedroom when sparks came out from the power cord and burned a deep black hole into the wood floor"
    frame = pd.DataFrame({
        "Manufacturer_norm": ["ACME"] * 5, "Brand_norm": ["ACME"] * 5,
        "Model_norm": ["MODEL 1", "MODEL 1", "UNKNOWN", "I DON T KNOW", "UNKNOWN"],
        "UPC_norm": [None] * 5, "recall_index": [None, 7, 7, None, None],
        "Report No.": [str(i) for i in range(5)],
        "Incident Description": ["unique text", "different text", base, base + " overnight", "unrelated"],
    })
    groups, buckets = related_reports(frame)
    assert len(set(groups[:4])) == 1
    assert groups[4] != groups[0]
    overlap = overlap_counts(buckets, [0, 2], [1, 3, 4])
    assert overlap["near incident"] == {"shared_keys": 1, "test_rows": 1}
    assert overlap["manufacturer"] == {"shared_keys": 1, "test_rows": 3}


def test_scorer_returns_finite_association_for_a_new_report():
    model = evaluation.fit_scorer(reports())
    X = prepare_features(reports(2).assign(Brand="new brand"))
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (2, 2)
    assert np.isfinite(probabilities).all()
    assert ((0 <= probabilities) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
