"""Data, linkage, split and model smoke tests using synthetic observations."""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from recallguard.api import app
from recallguard.data import (
    TARGET,
    feature_frame,
    load_incidents,
    load_recalls,
    manufacturer_key,
)
from recallguard.evaluation import partitions, product_groups
from recallguard.inference import IncidentRequest, load_model, predict, save_model
from recallguard.linkage import LinkageConfig, construct_labels, link, review_sample
from recallguard.training import train


@pytest.fixture(scope="module")
def historical():
    return pd.DataFrame(
        [
            {
                "report_id": f"T{i}",
                "report_date": datetime(2020, 1, 1) + timedelta(days=i),
                "product": f"Unique product variant {i}",
                "description": "smoke burst" if i % 3 == 0 else "loose fitting",
                "manufacturer": "Sample Corp.",
                "brand": "Sample",
                "category": "Home",
                "product_type": "Appliance",
                "model": f"M{i}",
                "upc": "",
                TARGET: int(i % 3 == 0),
            }
            for i in range(120)
        ]
    )


@pytest.fixture(scope="module")
def artifact(historical, tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "test.joblib"
    bundle = train(historical, "group", 42, "all", smoke=True)
    bundle["metadata"]["dataset"] = "synthetic.csv"
    save_model(bundle, path)
    return path


def test_incident_ingestion(tmp_path):
    path = tmp_path / "inc.csv"
    path.write_text(
        '"Disclaimer: sample export"\n'
        "Report No.,Report Date,Product Description,label_recalled\n"
        "A,invalid,toaster,1\nA,invalid,toaster,1\nB,2025-01-01,,0\n"
        "C,2025-01-01,cup,0\nC,2025-01-01,plate,0\n,2025-01-01,plate,0\n"
    )
    result = load_incidents(path, labeled=True)
    assert result.report_id.tolist() == ["A", "B"]
    assert pd.isna(result.loc[0, "report_date"])
    assert result.loc[1, "manufacturer"] == ""
    path.write_text("wrong,header\n1,2\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_incidents(path)


def test_invalid_label(tmp_path):
    path = tmp_path / "inc.csv"
    path.write_text(
        "Report No.,Report Date,Product Description,label_recalled\nA,bad,,unknown\n"
    )
    with pytest.raises(ValueError, match="binary"):
        load_incidents(path, labeled=True)


def test_linkage(tmp_path):
    source = tmp_path / "recalls.csv"
    source.write_text(
        "Recall Number,Date,Name of product,Manufacturers,Recall Heading\n"
        "R1,2025-03-01,Maple Trail lithium lantern,Maple Trail,Maple Trail recall\n"
        "R2,2024-01-01,Maple Trail lithium lantern,Maple Trail,Maple Trail recall\n"
        "R3,2025-03-01,Maple Trail camping lantern,Maple Trail,Maple Trail recall\n"
        "R4,invalid,Maple Trail lithium lantern,Maple Trail,Maple Trail recall\n"
    )
    recall = load_recalls(source)
    incidents = pd.DataFrame(
        [
            {
                "report_id": "I",
                "product": "Maple Trail lithium lantern",
                "manufacturer": "Maple Trail LLC",
                "brand": "Maple Trail",
                "report_date": pd.Timestamp("2025-01-01"),
            },
            {
                "report_id": "U",
                "product": "",
                "manufacturer": "unknown",
                "brand": "",
                "report_date": pd.NaT,
            },
        ]
    )
    audit = link(incidents, recall, LinkageConfig()).set_index("recall_id")
    assert audit.confidence.to_dict() == {
        "R1": "high",
        "R2": "rejected",
        "R3": "review",
        "R4": "rejected",
    }
    sample = review_sample(audit.reset_index(), incidents, recall, per_status=1)
    assert len(sample) == 3
    assert set(sample.confidence) == {"high", "review", "rejected"}
    assert "recall_product" in sample
    labels = construct_labels(incidents, audit.reset_index())
    assert labels[TARGET].iloc[0] == 1
    assert pd.isna(labels[TARGET].iloc[1])
    duplicate = audit.reset_index().iloc[[0]].assign(recall_id="R5")
    labels = construct_labels(incidents, pd.concat([audit.reset_index(), duplicate]))
    assert labels[TARGET].isna().all()


def test_features_and_input():
    assert manufacturer_key(" Maple Trail, LLC. ") == "maple trail"
    frame = feature_frame(
        pd.DataFrame(
            {"product": ["", "a " * 10000], TARGET: [0, 1], "recall_id": ["x", "y"]}
        )
    )
    assert frame.text.iloc[0] == "missingtext"
    assert len(frame.text.iloc[1]) < 16000
    assert TARGET not in frame and "recall_id" not in frame
    for invalid in (
        {"description": "  "},
        {"description": "a" * 16001},
        {"product": 42},
        {"foo": "extra"},
    ):
        with pytest.raises(ValidationError):
            IncidentRequest.model_validate(invalid)


@pytest.mark.parametrize("design", ["group", "chronological"])
def test_holdout_integrity(design, historical):
    frame = historical.copy()
    frame.loc[119, "model"] = frame.loc[0, "model"]
    groups = product_groups(frame)
    assert groups.iloc[0] == groups.iloc[-1]
    splits, groups = partitions(frame, design, seed=42)
    seen = set()
    for part in splits:
        present = set(groups.loc[part.index])
        assert not present.intersection(seen)
        seen.update(present)
    if design == "chronological":
        assert all(
            left.report_date.max() < right.report_date.min()
            for left, right in zip(splits, splits[1:])
        )


def test_train_reproducible_and_serialized(historical, artifact):
    restored = load_model(artifact)
    repeated = train(historical, "group", 42, "all", smoke=True)
    assert repeated["metadata"]["comparison"] == restored["metadata"]["comparison"]
    request = IncidentRequest(description="smoke burst", category="Unseen category")
    prediction = predict(restored, request)
    assert prediction == predict(repeated, request)
    assert 0 <= prediction.priority_score <= 1
    assert prediction.contributions


def test_api_without_model(monkeypatch):
    monkeypatch.delenv("RECALLGUARD_MODEL", raising=False)
    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "model_available": False,
        }
        assert client.get("/model-info").status_code == 503
        assert client.post("/predict", json={"product": "toaster"}).status_code == 503
        assert client.post("/predict", content="{").status_code == 422


def test_api_with_model(artifact, monkeypatch):
    monkeypatch.setenv("RECALLGUARD_MODEL", str(artifact))
    with TestClient(app) as client:
        assert client.get("/health").json()["model_available"]
        assert client.get("/model-info").json()["design"] == "group"
        body = {"description": "smoke burst", "category": "Unseen category"}
        response = client.post("/predict", json=body)
        assert response.status_code == 200
        assert (
            response.json()
            == predict(load_model(artifact), IncidentRequest(**body)).model_dump()
        )
        assert (
            client.post("/predict", json={"product": "too long " * 4000}).status_code
            == 422
        )


def test_streamlit_pages(monkeypatch, artifact):
    monkeypatch.setenv("RECALLGUARD_MODEL", str(artifact))
    entry = Path(__file__).resolve().parent.parent / "app.py"
    page = AppTest.from_file(str(entry), default_timeout=30).run()
    assert not page.exception
    for view in (
        "Dataset explorer",
        "Record linkage",
        "Explainability",
        "Model performance",
        "About",
        "Recall risk assessment",
    ):
        page.sidebar.radio[0].set_value(view).run()
        assert not page.exception
    page.text_area[0].set_value("Smoke burst")
    page.button[0].click().run()
    assert not page.exception
    assert len(page.metric) == 2


def test_description_identifier_linkage():
    date = pd.Timestamp("2025-04-01")
    recalls = pd.DataFrame(
        [
            {
                "recall_id": key,
                "product": "Countertop appliance",
                "heading": "",
                "manufacturer": maker,
                "recall_date": recall_date,
                "recall_description": description,
            }
            for key, maker, recall_date, description in [
                ("CORROBORATED", "Teal", date, "Affected models RG-431X"),
                ("AMBIGUOUS", "Different", date, "Affected models RG-431X"),
                ("PAST", "Teal", pd.Timestamp("2023-04-01"), "Affected models RG-431X"),
                ("UPC", "Different", date, "UPC 036000291452"),
            ]
        ]
    )
    incidents = pd.DataFrame(
        [
            {
                "report_id": key,
                "product": "",
                "brand": "",
                "manufacturer": "Teal LLC",
                "report_date": pd.Timestamp("2025-01-01"),
                "model": model,
                "upc": upc,
            }
            for key, model, upc in [
                ("MODEL", "RG431X", ""),
                ("GTIN", "", "0036000291452"),
                ("INVALID", "", "036000291453"),
            ]
        ]
    )
    matches = link(incidents, recalls, LinkageConfig(max_candidates=1))
    assert matches.set_index(["report_id", "recall_id"]).confidence.to_dict() == {
        ("MODEL", "CORROBORATED"): "high",
        ("MODEL", "AMBIGUOUS"): "review",
        ("MODEL", "PAST"): "rejected",
        ("GTIN", "UPC"): "high",
        ("INVALID", "CORROBORATED"): "rejected",
    }
    assert not matches.loc[matches.report_id.eq("INVALID"), "gtin_exact"].any()


def test_f1_selection_with_fielded_text(historical):
    frame = historical.assign(description="")
    bundle = train(frame, "group", 42, "fielded", smoke=True, selection="f1")
    metadata = bundle["metadata"]
    assert metadata["selection"] == "f1"
    report = metadata["comparison"][metadata["selected"]]
    assert 0 <= report["validation_f1"] <= 1
    names = bundle["pipeline"].named_steps["preprocess"].get_feature_names_out()
    assert any(name.startswith("product_word__") for name in names)
    assert all(not name.startswith("description_word__") for name in names)
