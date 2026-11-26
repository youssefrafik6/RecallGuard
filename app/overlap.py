"""Audit and group related reports using fields retained in the capstone CSV."""

from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

MISSING = {
    "", "UNKNOWN", "NONE", "N/A", "NA", "NOT AVAILABLE", "NOT APPLICABLE",
    "UNAVAILABLE", "NOT PROVIDED", "NO MODEL NUMBER", "I DON T KNOW",
    "DO NOT KNOW", "DON T KNOW", "UNK",
}


def _values(frame, column):
    return (frame[column].astype("string").fillna("").str.upper()
            .str.replace(r"\s+", " ", regex=True).str.strip())


def _normalize_text(series):
    return (series.astype("string").fillna("").str.casefold()
            .str.replace(r"[^\w]+", " ", regex=True).str.strip())


def related_reports(frame):
    """Return connected components and membership buckets for the overlap audit."""
    manufacturer = _values(frame, "Manufacturer_norm")
    brand = _values(frame, "Brand_norm")
    model = _values(frame, "Model_norm")
    upc = _values(frame, "UPC_norm")
    campaign = _values(frame, "recall_index")
    report = _values(frame, "Report No.")
    text = _normalize_text(frame["Incident Description"])
    buckets = defaultdict(lambda: defaultdict(list))

    for i in range(len(frame)):
        for kind, value in (
            ("report number", report.iloc[i]),
            ("recall candidate", campaign.iloc[i]),
            ("manufacturer", manufacturer.iloc[i]),
            ("brand", brand.iloc[i]),
        ):
            if value not in MISSING:
                buckets[kind][value].append(i)
        if model.iloc[i] not in MISSING:
            for kind, scope in (("manufacturer + model", manufacturer), ("brand + model", brand)):
                if scope.iloc[i] not in MISSING:
                    buckets[kind][(scope.iloc[i], model.iloc[i])].append(i)
        if upc.iloc[i].isdigit() and len(upc.iloc[i]) in (8, 12, 13, 14):
            buckets["UPC"][upc.iloc[i]].append(i)
        if text.iloc[i]:
            buckets["exact normalized incident"][text.iloc[i]].append(i)

    # Binary word-trigram cosine finds small edits in long narratives while
    # retaining digits. Batches keep the pairwise comparison memory bounded.
    eligible = np.flatnonzero(text.str.split().str.len().to_numpy() >= 12)
    near_pairs = []
    if len(eligible):
        vectorizer = CountVectorizer(ngram_range=(3, 3), binary=True, dtype=np.float64,
                                     token_pattern=r"(?u)\b\w+\b")
        matrix = normalize(vectorizer.fit_transform(text.iloc[eligible]))
        for start in range(0, len(eligible), 128):
            similarities = (matrix[start:start + 128] @ matrix.T).tocoo()
            for row, column in zip(similarities.row[similarities.data >= 0.9],
                                   similarities.col[similarities.data >= 0.9]):
                left, right = eligible[start + row], eligible[column]
                if left < right and text.iloc[left] != text.iloc[right]:
                    near_pairs.append((left, right))
    buckets["near incident"] = dict(enumerate(near_pairs))

    parent = list(range(len(frame)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for kind, groups in buckets.items():
        if kind in {"manufacturer", "brand"}:
            continue
        for members in groups.values():
            for index in members[1:]:
                parent[root(index)] = root(members[0])
    return np.array([root(i) for i in range(len(frame))]), dict(buckets)


def overlap_counts(buckets, train, test):
    """Count shared keys and distinct holdout rows touched by each rule."""
    train, test = set(train), set(test)
    counts = {}
    for kind, groups in buckets.items():
        shared = [test.intersection(rows) for rows in groups.values()
                  if train.intersection(rows) and test.intersection(rows)]
        counts[kind] = {"shared_keys": len(shared), "test_rows": len(set().union(*shared))}
    return counts
