"""Generate reviewable recall candidates using rare product token blocks."""

import re
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


MODEL_TOKEN = re.compile(r"(?<![a-z0-9])[a-z0-9]+(?:[-./][a-z0-9]+)*(?![a-z0-9])", re.I)
GTIN_TOKEN = re.compile(r"(?<![0-9])[0-9]{8,14}(?![0-9])")


def model_codes(text: str) -> set[str]:
    """Extract exact mixed letter/digit identifiers, including punctuated model codes."""
    codes = set()
    for token in MODEL_TOKEN.findall(text.casefold()):
        code = re.sub(r"[^a-z0-9]", "", token)
        if (
            5 <= len(code) <= 24
            and re.search(r"[a-z]", code)
            and re.search(r"[0-9]", code)
        ):
            codes.add(code)
    return codes


def gtin_codes(text: str) -> set[str]:
    """Recognize checksum-valid GTINs and normalize equivalent padding."""
    codes = set()
    for code in GTIN_TOKEN.findall(text):
        if len(code) not in {8, 12, 13, 14} or len(set(code)) == 1:
            continue
        weighted = sum(
            int(digit) * (3 if index % 2 else 1)
            for index, digit in enumerate(reversed(code))
        )
        if weighted % 10 == 0:
            codes.add(code.zfill(14))
    return codes


def link(
    incidents: pd.DataFrame, recalls: pd.DataFrame, config: LinkageConfig
) -> pd.DataFrame:
    """Score bounded candidate pairs and keep evidence for accepted and rejected matches."""
    catalog = {}
    blocks = defaultdict(list)
    model_blocks = defaultdict(list)
    gtin_blocks = defaultdict(list)
    for recall in recalls.itertuples(index=False):
        product = normalize(recall.product)
        product_tokens = tokens(product)
        heading_tokens = tokens(recall.heading)
        maker = manufacturer_key(recall.manufacturer)
        listing = " ".join(
            (recall.product, recall.heading, getattr(recall, "recall_description", ""))
        )
        for code in model_codes(listing):
            model_blocks[code].append(recall.recall_id)
        for code in gtin_codes(listing):
            gtin_blocks[code].append(recall.recall_id)
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
        models = {
            recall_id
            for code in model_codes(getattr(incident, "model", ""))
            for recall_id in model_blocks.get(code, ())
            if len(model_blocks[code]) <= config.max_block
        }
        gtins = {
            recall_id
            for code in gtin_codes(getattr(incident, "upc", ""))
            for recall_id in gtin_blocks.get(code, ())
        }
        candidates = votes.keys() | models | gtins
        offsets = {
            key: (catalog[key][0].recall_date - incident.report_date).days
            if pd.notna(catalog[key][0].recall_date) and pd.notna(incident.report_date)
            else None
            for key in candidates
        }
        ranked = sorted(
            candidates,
            key=lambda key: (
                key not in gtins,
                key not in models,
                offsets[key] is None or not 0 < offsets[key] <= config.horizon_days,
                -votes[key],
                key,
            ),
        )[: config.max_candidates]
        # Preserve every direct identifier candidate so competing matches stay visible.
        for recall_id in [*ranked, *sorted((models | gtins).difference(ranked))]:
            _, name, name_tokens, heading, recalled_maker = catalog[recall_id]
            days = offsets[recall_id]
            model_hit = recall_id in models
            gtin_hit = recall_id in gtins
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
            elif gtin_hit:
                status, reason = "high", "Exact checksum-valid GTIN"
            elif model_hit and (identity or brand_hit):
                status, reason = (
                    "high",
                    "Exact model code and manufacturer or brand agreement",
                )
            elif model_hit:
                status, reason = (
                    "review",
                    "Exact model code without corroborating identity",
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
                    model_hit,
                    gtin_hit,
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
        "model_exact",
        "gtin_exact",
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


def reference_coverage(
    audit: pd.DataFrame, reference: pd.DataFrame, horizon_days: int
) -> dict:
    """Measure agreement with earlier candidate links using independently recorded dates."""
    keys = ["report_id", "recall_id"]
    reference = reference.copy()
    if reference[keys].eq("").any().any() or reference.duplicated(keys).any():
        raise ValueError("Reference pairs must have unique, nonempty identifiers")
    dates = pd.to_datetime(reference["Date"], format="mixed", errors="raise")
    reports = pd.to_datetime(reference["Report Date"], format="mixed", errors="raise")
    reference["eligible"] = (dates - reports).dt.days.between(1, horizon_days)
    outcomes = reference.merge(
        audit[[*keys, "confidence"]], how="left", on=keys, validate="one_to_one"
    )
    return {
        name: {
            "reference_pairs": len(rows),
            "candidate_pairs": int(rows.confidence.notna().sum()),
            "review_pairs": int(rows.confidence.eq("review").sum()),
            "high_pairs": int(rows.confidence.eq("high").sum()),
        }
        for name, rows in (
            ("all", outcomes),
            ("date_eligible", outcomes.loc[outcomes.eligible]),
        )
    }
