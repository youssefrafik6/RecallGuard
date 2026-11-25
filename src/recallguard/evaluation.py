"""Split construction and metrics for retrospective proxy labels."""

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from recallguard.data import TARGET, manufacturer_key, normalize
from recallguard.linkage import gtin_codes


def product_groups(frame: pd.DataFrame, group_campaigns: bool = False) -> pd.Series:
    """Join overlapping UPC, manufacturer/model and exact product description proxies."""
    parents = list(range(len(frame)))
    seen = {}

    def root(index: int) -> int:
        while index != parents[index]:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, row in enumerate(frame.itertuples(index=False)):
        keys = []
        upc = normalize(row.upc).replace(" ", "")
        maker, model = manufacturer_key(row.manufacturer), normalize(row.model)
        product = normalize(row.product)
        if upc.isdigit() and len(upc) in (8, 12, 13, 14) and len(set(upc)) > 1:
            keys.append(("upc", upc))
        if maker and model:
            keys.append(("model", maker, model))
        if group_campaigns:
            campaign = getattr(row, "recall_campaign", "").strip()
            if campaign:
                keys.append(("recall_campaign", campaign))
            keys.extend(("gtin", code) for code in sorted(gtin_codes(upc)))
            if maker and model:
                keys.append(("model_compact", maker, model.replace(" ", "")))
        if len(product) >= 12:
            keys.append(("name", product))
        for key in keys:
            previous = seen.setdefault(key, index)
            parents[root(index)] = root(previous)
    return pd.Series([root(index) for index in range(len(frame))], index=frame.index)


def partitions(
    frame: pd.DataFrame, design: str, seed: int, group_campaigns: bool = False
) -> tuple[list, pd.Series]:
    """Return training, validation, calibration and test sets in that order."""
    groups = product_groups(frame, group_campaigns)
    ids = frame.index.to_numpy()
    if design == "random":
        remaining, test = train_test_split(
            ids, test_size=0.2, stratify=frame[TARGET], random_state=seed
        )
        remaining, calibration = train_test_split(
            remaining,
            test_size=0.25,
            stratify=frame.loc[remaining, TARGET],
            random_state=seed,
        )
        train, validation = train_test_split(
            remaining,
            test_size=1 / 3,
            stratify=frame.loc[remaining, TARGET],
            random_state=seed,
        )
        indices = [train, validation, calibration, test]
    elif design == "group":
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = [ids[test] for _, test in splitter.split(ids, frame[TARGET], groups)]
        indices = [np.concatenate(folds[:2]), *folds[2:]]
    elif design == "chronological":
        dated = frame.loc[frame.report_date.notna()].sort_values(
            "report_date", kind="stable"
        )
        if len(dated) < 5:
            raise ValueError("Chronological split needs at least five dated rows")
        dates = dated.report_date
        first, second, third = (
            dates.iloc[int(len(dates) * share)] for share in (0.4, 0.6, 0.8)
        )
        indices = [
            dated.index[dates.lt(first)],
            dated.index[dates.ge(first) & dates.lt(second)],
            dated.index[dates.ge(second) & dates.lt(third)],
            dated.index[dates.ge(third)],
        ]
        future_groups = set()
        for position in range(3, -1, -1):
            current = indices[position]
            indices[position] = current[~groups.loc[current].isin(future_groups)]
            future_groups.update(groups.loc[indices[position]])
    else:
        raise ValueError(f"Unknown design {design}")
    datasets = [frame.loc[part] for part in indices]
    if any(part[TARGET].nunique() != 2 for part in datasets):
        raise ValueError(f"{design} requires both classes in every partition")
    return datasets, groups


def score_metrics(target, scores, threshold: float) -> dict:
    predicted = scores >= threshold
    tn, fp, fn, tp = metrics.confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    order = np.argsort(-scores, kind="stable")
    actual = np.asarray(target)
    pr_p, pr_r, _ = metrics.precision_recall_curve(target, scores)
    fpr, tpr, _ = metrics.roc_curve(target, scores)
    top = {}
    for k in (10, 25, 50):
        subset = actual[order[:k]]
        top[str(k)] = {
            "k": len(subset),
            "precision": float(subset.mean()),
            "recall": float(subset.sum() / actual.sum()),
        }
    return {
        "precision": metrics.precision_score(target, predicted, zero_division=0),
        "recall": metrics.recall_score(target, predicted, zero_division=0),
        "f1": metrics.f1_score(target, predicted, zero_division=0),
        "macro_f1": metrics.f1_score(
            target, predicted, average="macro", zero_division=0
        ),
        "average_precision": metrics.average_precision_score(target, scores),
        "roc_auc": metrics.roc_auc_score(target, scores),
        "false_positive_rate": float(fp / (fp + tn)),
        "false_negative_rate": float(fn / (fn + tp)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "pr_curve": {"recall": pr_r.tolist(), "precision": pr_p.tolist()},
        "roc_curve": {
            "false_positive_rate": fpr.tolist(),
            "true_positive_rate": tpr.tolist(),
        },
        "top_k": top,
    }
