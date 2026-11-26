import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLS, REQUIRED_COLS, load_data, prepare_features, prepare_prediction
from risk import HAZARD_WORDS, clean_text, composite_risk_score, compute_severity_score, hazard_tokens, keyword_count


def test_load_data_preserves_upc_and_validates_schema(tmp_path):
    path = tmp_path / "reports.csv"
    frame = pd.DataFrame(columns=REQUIRED_COLS, index=range(2))
    frame["label_recalled"] = [0, 1]
    frame["UPC_norm"] = ["000123456789", None]
    frame.to_csv(path, index=False)
    assert load_data(path)["UPC_norm"].iloc[0] == "000123456789"
    frame.drop(columns=["Brand"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns: Brand"):
        load_data(path)
    frame["label_recalled"] = [0, None]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="binary classes"):
        load_data(path)


def test_text_cleaning_and_keyword_matching():
    assert clean_text("REDACTED http://t.example A 4 x 2 x 1 unit at 10 volts, ref 123.") == (
        "a dimension unit at measurement ref"
    )
    assert keyword_count("fire FIRE fireproof", HAZARD_WORDS) == 2
    assert "caught fire" in hazard_tokens("It CAUGHT FIRE.")
    assert "fire" not in hazard_tokens("fireproof")


def test_severity_preserves_weights_and_cap():
    severity, critical, high, moderate = compute_severity_score("Hospitalized with pain after it caught fire")
    assert (critical, high, moderate) == (1, 2, 1)
    assert severity == pytest.approx(0.70)
    assert compute_severity_score("death hospitalized amputation")[0] == 1.0
    assert compute_severity_score("fireproof")[0] == 0.0


def test_composite_bounds_and_reference_weights():
    values = np.linspace(0, 1, 21)
    assert all(0 <= composite_risk_score(a, s) <= 1 for a in values for s in values)
    assert composite_risk_score(1, 0) == pytest.approx(0.7)
    assert composite_risk_score(0, 1) == pytest.approx(0.6)
    assert composite_risk_score(0.8, 0.5) == pytest.approx(0.665)


@pytest.mark.parametrize("invalid", [-0.1, 1.1, np.nan, np.inf])
def test_composite_rejects_invalid_signals(invalid):
    with pytest.raises(ValueError, match="between 0 and 1"):
        composite_risk_score(invalid, 0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        composite_risk_score(0, invalid)


@pytest.mark.parametrize("values", [{}, {"Product Description": "  "}, {"Incident Description": None}, {"Incident Description": np.nan}, {"Incident Description": pd.NA}])
def test_prediction_requires_description(values):
    with pytest.raises(ValueError, match="at least a product description"):
        prepare_prediction(values)


def test_prediction_and_training_use_same_features_for_missing_inputs():
    values = {"Incident Description": " fire reported ", "Model Name or Number": " "}
    training = prepare_features(pd.DataFrame([values], columns=FEATURE_COLS))
    prediction = prepare_prediction(values)
    pd.testing.assert_frame_equal(prediction, training)
    assert prediction.iloc[0]["product_word_count"] == 0
    assert prediction.iloc[0]["incident_hazard_count"] == 1
    assert prediction.iloc[0]["has_model_info"] == 0
