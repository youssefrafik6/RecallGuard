"""RecallGuard local research dashboard."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from recallguard.data import TARGET, load_incidents
from recallguard.inference import LIMITATION, configured_model, global_features, predict
from recallguard.inference import IncidentRequest as Input
from recallguard.linkage import LinkageConfig, link

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="RecallGuard", page_icon="▣", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 3rem;}
    [data-testid="stMetric"] {border: 1px solid #768b9840; border-radius: 12px; padding: 1rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #768b9840;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def model(path: str, modified: int):
    return configured_model()


@st.cache_data
def dataset(path: Path):
    return load_incidents(path, labeled=True)


def spreadsheet_safe(value):
    """Keep formula-like text inert when a downloaded CSV opens in a spreadsheet."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


@st.cache_data
def candidate_demo():
    incidents = pd.DataFrame(
        [
            {
                "report_id": "DEMO-01",
                "product": "Maple Trail lithium lantern",
                "manufacturer": "Maple Trail LLC",
                "brand": "Maple Trail",
                "report_date": pd.Timestamp("2025-01-01"),
            }
        ]
    )
    recalls = pd.DataFrame(
        [
            {
                "recall_id": "DEMO-101",
                "product": "Maple Trail lithium lantern",
                "manufacturer": "Maple Trail",
                "heading": "Maple Trail lantern recall",
                "recall_date": pd.Timestamp("2025-02-01"),
            },
            {
                "recall_id": "DEMO-102",
                "product": "Maple Trail solar lantern",
                "manufacturer": "Maple Trail",
                "heading": "Maple Trail lantern recall",
                "recall_date": pd.Timestamp("2024-11-01"),
            },
        ]
    )
    return link(incidents, recalls, LinkageConfig())


path = os.getenv("RECALLGUARD_MODEL", "")
modified = Path(path).stat().st_mtime_ns if path and Path(path).is_file() else 0
bundle = model(path, modified)
st.sidebar.title("RecallGuard")
st.sidebar.caption("CPSC incident research")
page = st.sidebar.radio(
    "Explore",
    [
        "Overview",
        "Dataset explorer",
        "Recall risk assessment",
        "Model performance",
        "Record linkage",
        "Explainability",
        "About",
    ],
)
st.sidebar.caption("Local model loaded" if bundle else "No local model configured")
st.title(page if page != "Overview" else "Consumer product recall analysis")
st.caption(
    "Retrospective analysis of links between consumer incident reports and CPSC recalls."
)

if page in {"Overview", "Dataset explorer"}:
    try:
        data = dataset(ROOT / "data" / "training_data.csv")
    except (OSError, ValueError):
        st.warning("The historical training CSV is unavailable.")
        st.stop()

if page == "Overview":
    left, middle, right = st.columns(3)
    left.metric("Labeled observations", f"{len(data):,}")
    middle.metric("Recall-linked labels", f"{data[TARGET].sum():,}")
    right.metric("Linked share", f"{data[TARGET].mean():.1%}")
    st.subheader("From public reports to review priorities")
    st.write(
        "CPSC incident reports and recall listings have no shared product identifier. "
        "RecallGuard shows how candidate links create labels and how a classifier ranks historical associations."
    )
    st.info(
        "The original capstone cites about 39,946 incident reports and 9,666 recall records. "
        "The figures above describe its checked-in labeled subset."
    )
    if bundle:
        metadata = bundle["metadata"]
        report = metadata["comparison"][metadata["selected"]]
        st.subheader("Loaded model holdout")
        st.write(f"{metadata['selected']} · {metadata['design']} split")
        left, right = st.columns(2)
        left.metric("F1", f"{report['f1']:.3f}")
        right.metric("Average precision", f"{report['average_precision']:.3f}")
    st.warning(
        "A high score indicates similarity to historical linked observations. "
        "Future recall and actual hazard require separate evidence."
    )

elif page == "Dataset explorer":
    filters, results = st.columns([1, 3])
    with filters:
        categories = st.multiselect("Category", sorted(set(data.category) - {""}))
        maker = st.text_input("Manufacturer contains")
        labels = st.multiselect("Historical linkage label", [0, 1], default=[0, 1])
    view = data.loc[data[TARGET].isin(labels)]
    if categories:
        view = view.loc[view.category.isin(categories)]
    if maker:
        view = view.loc[view.manufacturer.str.contains(maker, case=False, regex=False)]
    with results:
        st.subheader(f"{len(view):,} matching reports")
        st.bar_chart(view.category.replace("", "Unspecified").value_counts().head(12))
        st.bar_chart(
            view.manufacturer.replace("", "Unspecified").value_counts().head(12)
        )
        st.line_chart(view.groupby(view.report_date.dt.year).size())
        st.bar_chart(view[TARGET].value_counts())
        columns = [
            "report_date",
            "category",
            "manufacturer",
            "product",
            "description",
            TARGET,
        ]
        st.dataframe(view[columns], hide_index=True, width="stretch")
        st.download_button(
            "Download filtered reports",
            view[columns].map(spreadsheet_safe).to_csv(index=False),
            "recallguard.csv",
            "text/csv",
        )

elif page == "Record linkage":
    st.info("Synthetic example. The product and recall identifiers are fictional.")
    st.write(
        "DEMO-01 reports a Maple Trail lithium lantern. Two candidates share product terms. "
        "The solar lantern listing is rejected because its announcement precedes the report."
    )
    st.dataframe(candidate_demo(), hide_index=True, width="stretch")
    audit_path = ROOT / "data/processed/review_sample.csv"
    if audit_path.is_file():
        st.subheader("Local CPSC candidate audit")
        preview = pd.read_csv(audit_path)
        status = st.selectbox(
            "Candidate confidence", ["All", "high", "review", "rejected"]
        )
        st.caption(
            "Showing up to 30 high-scoring cases per status. This sample is for review."
        )
        st.dataframe(
            preview if status == "All" else preview.loc[preview.confidence.eq(status)],
            hide_index=True,
            width="stretch",
        )

elif page == "About":
    st.subheader("Youssef Rafik · Southern Connecticut State University")
    st.write(
        "Bachelor of Science in Data Science capstone, Spring 2026. The original project "
        "connected public incident reports to official recall listings and compared NLP classifiers. "
        "The portfolio upgrade adds reproducible split comparisons, local model artifacts and a shared API."
    )
    st.write("Python · pandas · scikit-learn · XGBoost · Streamlit · FastAPI")
    st.info(
        "Missed links, report availability, catalog coverage, and product identity ambiguity "
        "limit how far the results generalize."
    )

elif bundle is None:
    st.info(
        "Train a model with the CLI and set RECALLGUARD_MODEL to its trusted bundle before launching."
    )

elif page == "Recall risk assessment":
    with st.form("assessment"):
        product = st.text_input("Product description", max_chars=16000)
        description = st.text_area("Incident description", max_chars=16000)
        left, right = st.columns(2)
        category = left.text_input("Product category", max_chars=240)
        brand = right.text_input("Brand", max_chars=240)
        manufacturer = left.text_input("Manufacturer", max_chars=240)
        product_type = right.text_input("Product type", max_chars=240)
        submitted = st.form_submit_button("Assess review priority", type="primary")
    if submitted:
        try:
            request = Input(
                product=product,
                description=description,
                category=category,
                brand=brand,
                manufacturer=manufacturer,
                product_type=product_type,
            )
        except ValidationError:
            st.error("Enter a product description or an incident description.")
        else:
            result = predict(bundle, request)
            left, right = st.columns(2)
            left.metric("Review priority", result.priority)
            right.metric("Association score", f"{result.priority_score:.3f}")
            st.caption(LIMITATION)
            if result.contributions:
                st.bar_chart(
                    pd.DataFrame([item.model_dump() for item in result.contributions]),
                    x="feature",
                    y="effect",
                    horizontal=True,
                )
                st.caption(
                    "Contributions describe the model's base score. They do not establish causes."
                )

elif page == "Model performance":
    metadata = bundle["metadata"]
    comparisons = metadata["comparison"]
    st.write(
        f"{metadata['dataset']} · {metadata['design']} holdout · seed {metadata['seed']}"
    )
    criterion = "F1" if metadata.get("selection") == "f1" else "average precision"
    st.caption(
        f"Selected using validation {criterion}: {metadata['selected']}. "
        "Positive labels are rare and the dataset was sampled."
    )
    fields = [
        "variant",
        "precision",
        "recall",
        "f1",
        "average_precision",
        "roc_auc",
        "brier",
    ]
    st.dataframe(pd.DataFrame(comparisons).T[fields], width="stretch")
    family = st.selectbox(
        "Inspect model",
        list(comparisons),
        index=list(comparisons).index(metadata["selected"]),
    )
    report = comparisons[family]
    st.dataframe(
        pd.DataFrame(
            report["confusion_matrix"],
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
    )
    st.dataframe(
        pd.DataFrame(report["top_k"]).T.reset_index(drop=True), hide_index=True
    )
    st.caption(f"Validation-selected high-priority cutoff: {report['threshold']:.3f}")
    if report.get("category_analysis"):
        with st.expander("Held-out metrics by category"):
            st.caption("Slices with at least 20 test examples and both labels.")
            st.dataframe(report["category_analysis"], hide_index=True)
    for title, field, x in [
        ("Precision-recall", "pr_curve", "recall"),
        ("Receiver operating characteristic", "roc_curve", "false_positive_rate"),
        ("Reliability", "reliability", "mean_score"),
        ("Validation threshold trade-off", "threshold_analysis", "threshold"),
    ]:
        st.subheader(title)
        curve = pd.DataFrame(report[field])
        if field == "reliability":
            curve["perfect_calibration"] = curve[x]
        st.line_chart(curve, x=x)

elif page == "Explainability":
    values = pd.DataFrame(global_features(bundle))
    st.caption(
        "Coefficients or tree feature importance describe how the selected estimator uses inputs. "
        "Individual contributions are available after scoring a report."
    )
    if values.empty:
        st.info("The current estimator does not expose global feature weights.")
    else:
        st.bar_chart(values, x="feature", y="weight", horizontal=True)
