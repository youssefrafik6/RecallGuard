"""CPSC ingestion and deterministic normalization shared by training and linkage."""

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
TARGET = "label_recalled"
INCIDENT = {
    "Report No.": "report_id",
    "Report Date": "report_date",
    "Product Description": "product",
    "Incident Description": "description",
    "Manufacturer / Importer / Private Labeler Name": "manufacturer",
    "Brand": "brand",
    "Product Category": "category",
    "Product Type": "product_type",
    "Model Name or Number": "model",
    "UPC": "upc",
    "recall_index": "recall_campaign",
    "Publication Date": "publication_date",
    "best_confidence": "historical_confidence",
    "match_rule": "historical_match_rule",
}
RECALL = {
    "Recall Number": "recall_id",
    "Date": "recall_date",
    "Name of product": "product",
    "Recall Heading": "heading",
    "Description": "recall_description",
    "Manufacturers": "manufacturer",
    "Product Safety Warning Number": "warning_id",
}
FEATURES = (
    "product",
    "description",
    "category",
    "brand",
    "manufacturer",
    "product_type",
)
PLACEHOLDERS = {"unknown", "none", "nan", "na", "n a", "not applicable", "unspecified"}


def normalize(value: str) -> str:
    """Normalize case, spacing and punctuation while retaining Unicode letters."""
    value = unicodedata.normalize("NFKC", value).casefold()
    result = " ".join(re.findall(r"[^\W_]+", value))
    return "" if result in PLACEHOLDERS else result


def manufacturer_key(value: str) -> str:
    """Strip common legal suffixes for identity comparisons."""
    tokens = normalize(value).split()
    while tokens and tokens[-1] in {
        "inc",
        "llc",
        "ltd",
        "corp",
        "corporation",
        "limited",
    }:
        tokens.pop()
    return " ".join(tokens)


def read_export(
    path: Path, mapping: dict[str, str], required: set[str]
) -> pd.DataFrame:
    """Read the supported exports, including the CPSC incident disclaimer preamble."""
    with path.open(encoding="utf-8-sig", errors="replace") as source:
        skip = int(source.readline().lstrip('"').startswith("Disclaimer:"))
    headers = pd.read_csv(
        path, skiprows=skip, nrows=0, encoding_errors="replace"
    ).columns
    missing = required.difference(headers)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
    frame = pd.read_csv(
        path,
        usecols=lambda name: name in mapping or name == TARGET,
        dtype=str,
        keep_default_na=False,
        skiprows=skip,
        encoding_errors="replace",
        on_bad_lines="warn",
    ).rename(columns=mapping)
    for field in mapping.values():
        if field not in frame:
            frame[field] = ""
        frame[field] = frame[field].fillna("").str.strip()
    for field in ("report_date", "recall_date", "publication_date"):
        if field in frame:
            frame[field] = pd.to_datetime(frame[field], format="mixed", errors="coerce")
            LOGGER.info(
                "%s has %d missing or invalid %s values",
                path.name,
                frame[field].isna().sum(),
                field,
            )
    LOGGER.info("Read %d observations from %s", len(frame), path.name)
    return frame


def unique_records(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """Collapse exact duplicates and discard ambiguous repeated identifiers."""
    unique = frame.drop_duplicates()
    keep = unique[key].ne("") & ~unique[key].duplicated(keep=False)
    LOGGER.info(
        "Excluded %d missing, duplicate or conflicting %s rows",
        len(frame) - keep.sum(),
        key,
    )
    return unique.loc[keep].reset_index(drop=True)


def load_incidents(path: Path, labeled: bool = False) -> pd.DataFrame:
    required = {"Report No.", "Report Date", "Product Description"}
    if labeled:
        required.add(TARGET)
    frame = unique_records(read_export(path, INCIDENT, required), "report_id")
    if labeled:
        if not frame[TARGET].isin(["0", "1"]).all():
            raise ValueError("Expected binary historical labels 0 and 1")
        frame[TARGET] = frame[TARGET].astype(int)
    return frame


def load_recalls(path: Path) -> pd.DataFrame:
    frame = read_export(path, RECALL, {"Recall Number", "Date", "Name of product"})
    return unique_records(frame, "recall_id")


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Allowlist fields and derive nonnegative text lengths for every estimator."""
    features = frame.reindex(columns=FEATURES).fillna("").astype(str)
    for field in FEATURES:
        cleaner = manufacturer_key if field == "manufacturer" else normalize
        features[field] = features[field].str.slice(0, 16000).map(cleaner)
    features["word_count"] = features["description"].str.split().str.len()
    features["text"] = (features["product"] + " " + features["description"]).str.strip()
    features["text"] = features["text"].replace("", "missingtext")
    return features
