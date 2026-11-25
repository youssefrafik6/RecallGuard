"""Generate reviewable recall candidates using rare product token blocks."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

import pandas as pd

from recallguard.data import TARGET, manufacturer_key, normalize

STOP = set(
    "the and for with from model product recall recalls its are that this consumer".split()
)


@dataclass(frozen=True)
class LinkageConfig:
    """Thresholds and limits for an unvalidated review heuristic."""

    high: float = 0.82
    review: float = 0.57
    max_block: int = 75
    max_candidates: int = 18
    horizon_days: int = 365

    def __post_init__(self) -> None:
        if not 0 <= self.review < self.high <= 1:
            raise ValueError("Require 0 <= review < high <= 1")
        if min(self.max_block, self.max_candidates, self.horizon_days) < 1:
            raise ValueError("Block, candidate and horizon limits must be positive")


def tokens(text: str) -> set[str]:
    return {
        word for word in normalize(text).split() if len(word) >= 3 and word not in STOP
    }


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def link(
    incidents: pd.DataFrame, recalls: pd.DataFrame, config: LinkageConfig
) -> pd.DataFrame:
    """Score bounded candidate pairs and keep evidence for accepted and rejected matches."""
    catalog = {}
    blocks = defaultdict(list)
    for recall in recalls.itertuples(index=False):
        product = normalize(recall.product)
        product_tokens = tokens(product)
        heading_tokens = tokens(recall.heading)
        maker = manufacturer_key(recall.manufacturer)
        catalog[recall.recall_id] = (
            recall,
            product,
            product_tokens,
            heading_tokens,
            maker,
        )
        for token in product_tokens | heading_tokens | tokens(maker):
            blocks[token].append(recall.recall_id)

    rows = []
    for incident in incidents.itertuples(index=False):
        product = normalize(incident.product)
        product_tokens = tokens(product)
        maker = manufacturer_key(incident.manufacturer)
        brand = tokens(incident.brand)
        votes = Counter(
            candidate
            for token in product_tokens | brand | tokens(maker)
            if len(blocks.get(token, ())) <= config.max_block
            for candidate in blocks.get(token, ())
        )
        ranked = sorted(votes, key=lambda item: (-votes[item], item))[
            : config.max_candidates
        ]
        for recall_id in ranked:
            recall, name, name_tokens, heading, recalled_maker = catalog[recall_id]
            date_known = pd.notna(recall.recall_date) and pd.notna(incident.report_date)
            days = (
                (recall.recall_date - incident.report_date).days if date_known else None
            )
            overlap = jaccard(product_tokens, name_tokens)
            fuzzy = (
                SequenceMatcher(None, product, name).ratio()
                if product and name
                else 0.0
            )
            identity = float(bool(maker and maker == recalled_maker))
            brand_hit = float(bool(brand and brand <= (name_tokens | heading)))
            score = 0.50 * overlap + 0.30 * fuzzy + 0.15 * identity + 0.05 * brand_hit
            if days is None or not 0 < days <= config.horizon_days:
                status, reason = (
                    "rejected",
                    "Date missing or outside future linkage window",
                )
            elif not product_tokens & name_tokens:
                status, reason = "rejected", "No shared product-name tokens"
            elif score >= config.high and (identity or brand_hit):
                status, reason = (
                    "high",
                    "Product similarity and manufacturer or brand agreement",
                )
            elif score >= config.review:
                status, reason = "review", "Product similarity requires adjudication"
            else:
                status, reason = "rejected", "Similarity below review threshold"
            rows.append(
                (
                    incident.report_id,
                    recall_id,
                    days,
                    votes[recall_id],
                    overlap,
                    fuzzy,
                    identity,
                    brand_hit,
                    score,
                    status,
                    reason,
                )
            )
    columns = [
        "report_id",
        "recall_id",
        "days_to_recall",
        "block_votes",
        "token_overlap",
        "name_similarity",
        "manufacturer_exact",
        "brand_in_recall",
        "score",
        "confidence",
        "reason",
    ]
    return pd.DataFrame.from_records(rows, columns=columns)


def construct_labels(incidents: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Assign a proxy positive only when exactly one candidate has high confidence."""
    high = audit.loc[audit["confidence"].eq("high"), ["report_id", "recall_id"]]
    unique = high.loc[~high["report_id"].duplicated(keep=False)]
    result = incidents.merge(unique, how="left", on="report_id", validate="one_to_one")
    result[TARGET] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result.loc[result["recall_id"].notna(), TARGET] = 1
    return result


def review_sample(
    audit: pd.DataFrame,
    incidents: pd.DataFrame,
    recalls: pd.DataFrame,
    per_status: int = 30,
) -> pd.DataFrame:
    """Provide high-scoring cases from each status, with product names for local review."""
    subsets = [
        audit.loc[audit["confidence"].eq(status)].nlargest(per_status, "score")
        for status in ("high", "review", "rejected")
    ]
    sample = pd.concat(subsets, ignore_index=True)
    return sample.merge(
        incidents[["report_id", "product", "report_date"]].rename(
            columns={"product": "incident_product"}
        ),
        on="report_id",
        validate="many_to_one",
    ).merge(
        recalls[["recall_id", "product", "recall_date"]].rename(
            columns={"product": "recall_product"}
        ),
        on="recall_id",
        validate="many_to_one",
    )
