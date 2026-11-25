"""Aggregate drift and proxy-label diagnostics for the chronological split."""

import pandas as pd

from recallguard.data import TARGET, feature_frame
from recallguard.evaluation import partitions


def distribution_distance(reference: pd.Series, current: pd.Series) -> float:
    """Total variation ranges from zero to one for empirical distributions."""
    before = reference.value_counts(normalize=True)
    after = current.value_counts(normalize=True)
    return float(before.sub(after, fill_value=0).abs().sum() / 2)


def audit_temporal(
    frame: pd.DataFrame, group_campaigns: bool, reconstructed: set[str] | None = None
) -> dict:
    """Summarize covariate and label differences without displaying incident text."""
    splits, _ = partitions(frame, "chronological", 42, group_campaigns)
    training, _, _, test = splits
    train_features = feature_frame(training)
    earlier = pd.concat(splits[:-1])
    vocabulary = set(train_features.text.str.split().explode())
    results = {}
    for name, rows in zip(("train", "validation", "calibration", "test"), splits):
        features = feature_frame(rows)
        tokens = features.text.str.split().explode()
        positives = rows.loc[rows[TARGET].eq(1)]
        delay = (rows.publication_date - rows.report_date).dt.days
        result = {
            "rows": len(rows),
            "positives": len(positives),
            "positive_rate": float(rows[TARGET].mean()),
            "start": rows.report_date.min().date().isoformat(),
            "end": rows.report_date.max().date().isoformat(),
            "token_occurrence_oov_rate": float((~tokens.isin(vocabulary)).mean()),
            "unseen_nonblank_manufacturer_rows": int(
                (
                    features.manufacturer.ne("")
                    & ~features.manufacturer.isin(train_features.manufacturer)
                ).sum()
            ),
            "category_total_variation": distribution_distance(
                train_features.category, features.category
            ),
            "manufacturer_total_variation": distribution_distance(
                train_features.manufacturer, features.manufacturer
            ),
            "blank_model_rows": int(rows.model.eq("").sum()),
            "blank_upc_rows": int(rows.upc.eq("").sum()),
            "recall_term_rows": int(
                features.description.str.contains(r"\brecall(?:s|ed|ing)?\b").sum()
            ),
            "publication_lag_median_days": float(delay.median())
            if delay.notna().any()
            else None,
            "positive_campaigns": int(
                positives.recall_campaign.replace("", pd.NA).nunique()
            ),
            "positive_reports_in_top_three_campaigns": int(
                positives.loc[positives.recall_campaign.ne(""), "recall_campaign"]
                .value_counts()
                .head(3)
                .sum()
            ),
            "positive_link_rules": {
                str(k): int(v)
                for k, v in positives.historical_match_rule.value_counts().items()
            },
            "positive_confidence": {
                str(k): int(v)
                for k, v in positives.historical_confidence.value_counts().items()
            },
        }
        if reconstructed is not None:
            linked = rows.report_id.isin(reconstructed)
            result["reconstruction_positive_old_positive"] = int(
                (linked & rows[TARGET].eq(1)).sum()
            )
            result["reconstruction_positive_old_negative"] = int(
                (linked & rows[TARGET].eq(0)).sum()
            )
        results[name] = result
    overlap = test.recall_campaign.ne("") & test.recall_campaign.isin(
        earlier.recall_campaign
    )
    positive_overlap = (
        test[TARGET].eq(1)
        & test.recall_campaign.ne("")
        & test.recall_campaign.isin(
            earlier.loc[earlier[TARGET].eq(1), "recall_campaign"]
        )
    )
    return {
        "group_campaigns": group_campaigns,
        "test_positives_sharing_earlier_positive_campaign": int(positive_overlap.sum()),
        "partitions": results,
        "test_rows_sharing_earlier_campaign": int(overlap.sum()),
        "excluded_rows": len(frame) - sum(map(len, splits)),
    }
