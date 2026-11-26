import json
from hashlib import file_digest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from evaluation import category_insights, develop_scorer
from features import DATA_PATH, load_data, prepare_prediction
from risk import (
    SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MODERATE,
    composite_risk_score, compute_severity_score, hazard_tokens,
)

# SECTION 2: STREAMLIT PAGE CONFIGURATION
# This sets the browser tab title, icon, and layout.
st.set_page_config(
    page_title="RecallGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# SECTION 3: CUSTOM CSS STYLING
# This makes the app look polished and presentation-ready.
# I use custom fonts, dark metric cards, colored risk boxes, etc.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');

/* Apply DM Sans to main text elements */
.stApp, .stMarkdown, .stMarkdown p, .stMarkdown li,
div[data-testid="stText"], .stCaption, .stSelectbox label,
.stTextInput label, .stTextArea label, .stRadio label {
    font-family: 'DM Sans', sans-serif !important;
}

/* Use Space Mono (monospace) for headings */
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

/* Dark gradient sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stRadio label {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem;
    padding: 0.45rem 0;
}

/* Dark styled metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.6rem !important;
}

/* Risk score result boxes (green/orange/red) */
.risk-box {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    margin: 12px 0;
}
.risk-box h2 {
    margin: 0 0 4px 0;
    font-size: 2.6rem;
    font-family: 'Space Mono', monospace;
}
.risk-box p { margin: 0; font-size: 0.95rem; opacity: 0.85; }
.risk-low { background: linear-gradient(135deg, #065f46, #047857); color: #d1fae5; }
.risk-moderate { background: linear-gradient(135deg, #92400e, #b45309); color: #fef3c7; }
.risk-high { background: linear-gradient(135deg, #991b1b, #dc2626); color: #fee2e2; }

/* Hero banner on the dashboard */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border-radius: 20px;
    padding: 48px 40px;
    text-align: center;
    margin-bottom: 28px;
    border: 1px solid #334155;
    box-shadow: 0 8px 40px rgba(0,0,0,0.2);
}
.hero h1 { color: #f8fafc; font-size: 2.8rem; margin-bottom: 8px; }
.hero p { color: #94a3b8; font-size: 1.1rem; max-width: 640px; margin: 0 auto; }

/* Colored keyword chips for category insights */
.kw-chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    margin: 3px 4px;
    font-family: 'Space Mono', monospace;
    font-weight: 500;
}
.kw-pos { background: #fee2e2; color: #991b1b; }
.kw-neg { background: #d1fae5; color: #065f46; }

/* Small uppercase section labels */
.section-header {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    margin-bottom: 12px;
    font-family: 'Space Mono', monospace;
    border-bottom: 2px solid #1e293b;
    padding-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_data(path, modified_ns):
    return load_data(path)


@st.cache_resource(show_spinner="Fitting the historical association model…")
def cached_scorer(frame):
    return develop_scorer(frame)


@st.cache_data(show_spinner=False)
def cached_categories(frame):
    return category_insights(cached_scorer(frame))


@st.cache_data(show_spinner=False)
def cached_benchmark(data_path, report_path, data_modified_ns, report_modified_ns):
    with data_path.open("rb") as data_file:
        digest = file_digest(data_file, "sha256").hexdigest()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["metadata"]["dataset_sha256"] != digest:
        raise ValueError("The benchmark was generated from a different dataset.")
    return report


try:
    df = cached_data(DATA_PATH, DATA_PATH.stat().st_mtime_ns)
except (OSError, ValueError) as error:
    st.error(f"Could not load the bundled dataset. {error}")
    st.stop()


# SECTION 10: SIDEBAR NAVIGATION
st.sidebar.markdown("## RecallGuard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Dataset Insights", "Category Insights", "Risk Scorer", "Model Details"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Capstone Project  ·  Youssef Rafik\n\n"
    "Southern Connecticut State University\n\n"
    "Data Science  ·  Spring 2026"
)

if page == "Dataset Insights":

    st.markdown("## Dataset Insights")

    # Summary stats
    total = len(df)
    recalled = int(df["label_recalled"].sum())
    not_recalled = total - recalled
    st.caption(f"{total:,} incident reports  |  "
               f"{recalled:,} recall-linked labels  |  "
               f"{not_recalled:,} unlinked labels")

    col1, col2 = st.columns(2)

    # Left column: class balance bar chart
    with col1:
        st.markdown(
            '<div class="section-header">Class Balance</div>',
            unsafe_allow_html=True,
        )

        counts = df["label_recalled"].value_counts().sort_index()
        fig_balance = go.Figure(go.Bar(
            x=["Unlinked label (0)", "Recall-linked label (1)"],
            y=counts.values,
            marker_color=["#334155", "#3b82f6"],
            text=counts.values,
            textposition="outside",
            textfont=dict(family="Space Mono", size=14),
        ))
        fig_balance.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=True, gridcolor="#1e293b"),
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig_balance, width="stretch")

    # Right column: recall rate by category
    with col2:
        st.markdown(
            '<div class="section-header">Categories by Positive Linkage Rate</div>',
            unsafe_allow_html=True,
        )

        # Group by category and calculate mean recall rate
        cat_stats = df.groupby("Product Category")["label_recalled"].agg(
            ["mean", "count"]
        )
        # Only show categories with at least 20 rows
        cat_stats = cat_stats[cat_stats["count"] >= 20]
        cat_stats = cat_stats.sort_values("mean", ascending=True)

        fig_categories = go.Figure(go.Bar(
            y=cat_stats.index,
            x=cat_stats["mean"],
            orientation="h",
            marker_color="#3b82f6",
            text=[f"{v:.1%}" for v in cat_stats["mean"]],
            textposition="outside",
            textfont=dict(family="Space Mono", size=11),
        ))
        fig_categories.update_layout(
            height=max(320, len(cat_stats) * 28),
            margin=dict(l=20, r=50, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#1e293b", tickformat=".0%"),
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig_categories, width="stretch")

    # Expandable data preview
    with st.expander("Preview data (first 50 rows)"):
        preview_cols = [
            "Product Description", "Product Category",
            "Brand", "Incident Description", "label_recalled",
        ]
        existing_cols = [c for c in preview_cols if c in df.columns]
        st.dataframe(
            df[existing_cols].head(50),
            width="stretch",
            height=350,
        )


# PAGE 3: RISK SCORER
# The user enters product and incident details, and the app returns
# a composite risk score that blends:
#   1. The model’s historical association score
#   2. A hazard severity signal (from how dangerous the text sounds)

elif page == "Category Insights":

    st.markdown("## Category Insights")
    st.caption(
        "Explore recall patterns and key risk terms within each "
        "product category."
    )

    cat_results, cat_keywords = cached_categories(df)
    st.caption("Metrics summarize the scorer’s logistic regression predictions on the "
               "development folds by category. Keyword models use development records only.")
    available_cats = sorted(cat_keywords)

    if not available_cats:
        st.info("Not enough data for category-specific keyword analysis.")
        st.stop()

    # Dropdown to pick a category
    selected_category = st.selectbox(
        "Select a category", available_cats, index=0
    )

    if selected_category:

        # Get this category's results and keywords
        cr = cat_results[selected_category]
        kw = cat_keywords[selected_category]

        # Show 4 metric cards for this category
        st.markdown("")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Rows", f"{cr.get('rows', 0):,}")
        mc2.metric("Positive Linkage Rate", f"{cr.get('positive_rate', 0):.1%}")
        mc3.metric("Development CV F1", f"{cr.get('f1', 0):.3f}")
        mc4.metric("Development CV Recall", f"{cr.get('recall', 0):.3f}")

        st.markdown("")

        # Two columns: positive keywords on left, negative on right
        col_pos, col_neg = st.columns(2)

        # Left column: keywords that push TOWARD recall
        with col_pos:
            st.markdown(
                '<div class="section-header">'
                'Keywords Associated With Linked Labels</div>',
                unsafe_allow_html=True,
            )

            if kw.get("positive"):
                # Show keywords as colored chips
                chips_html = ""
                for word, coef in kw["positive"]:
                    chips_html += (
                        f'<span class="kw-chip kw-pos">'
                        f'{word} ({coef:.2f})</span>'
                    )
                st.markdown(chips_html, unsafe_allow_html=True)

                # Also show as a bar chart
                pos_df = pd.DataFrame(
                    kw["positive"], columns=["keyword", "coefficient"]
                )
                fig_pos = go.Figure(go.Bar(
                    x=pos_df["coefficient"],
                    y=pos_df["keyword"],
                    orientation="h",
                    marker_color="#ef4444",
                ))
                fig_pos.update_layout(
                    height=max(280, len(pos_df) * 22),
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(autorange="reversed"),
                    xaxis=dict(showgrid=True, gridcolor="#1e293b"),
                    font=dict(family="DM Sans", size=11),
                )
                st.plotly_chart(fig_pos, width="stretch")
            else:
                st.info("Not enough data for keyword analysis.")

        # Right column: keywords that push AWAY from recall
        with col_neg:
            st.markdown(
                '<div class="section-header">'
                'Keywords Associated With Unlinked Labels</div>',
                unsafe_allow_html=True,
            )

            if kw.get("negative"):
                # Show keywords as colored chips
                chips_html = ""
                for word, coef in kw["negative"]:
                    chips_html += (
                        f'<span class="kw-chip kw-neg">'
                        f'{word} ({coef:.2f})</span>'
                    )
                st.markdown(chips_html, unsafe_allow_html=True)

                # Also show as a bar chart
                neg_df = pd.DataFrame(
                    kw["negative"], columns=["keyword", "coefficient"]
                )
                fig_neg = go.Figure(go.Bar(
                    x=neg_df["coefficient"],
                    y=neg_df["keyword"],
                    orientation="h",
                    marker_color="#10b981",
                ))
                fig_neg.update_layout(
                    height=max(280, len(neg_df) * 22),
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(autorange="reversed"),
                    xaxis=dict(showgrid=True, gridcolor="#1e293b"),
                    font=dict(family="DM Sans", size=11),
                )
                st.plotly_chart(fig_neg, width="stretch")
            else:
                st.info("Not enough data for keyword analysis.")

        # Expandable explanation of coefficients
        with st.expander("How to read the keyword coefficients"):
            st.markdown("""
Coefficients come from a Logistic Regression model trained only on
development text within this category. Positive values increase the
model score for the linked label. Negative values decrease it.
The coefficients describe associations in the category’s TF-IDF features.
            """)
elif page == "Risk Scorer":

    st.markdown("## Risk Scorer")
    st.caption(
        "Enter product and incident details to prioritize reports for review. "
        "The combined score ranges from 0 to 1. Future-recall probabilities require "
        "prospective validation and calibration. Official safety assessments come from the CPSC."
    )

    # --- Professor-requested feature: load examples from the dataset ---
    # Initialize session_state keys for pre-filling the form fields.
    # These get set when the user clicks "Load Random Example" or picks
    # a record, and cleared when they click "Clear Fields."
    field_keys = [
        "ex_product_desc", "ex_incident_desc", "ex_brand",
        "ex_manufacturer", "ex_model_num", "ex_category",
        "ex_subcategory", "ex_type",
    ]
    for fk in field_keys:
        if fk not in st.session_state:
            st.session_state[fk] = ""

    # --- Professor-requested feature: browse and load examples from dataset ---
    # Step 1: pick a category, Step 2: see product titles, Step 3: select one
    # This auto-fills the form fields below so the user doesn't have to copy/paste.

    browse_categories = sorted(df["Product Category"].dropna().unique().tolist())

    br1, br2 = st.columns([1, 2])

    with br1:
        browse_cat = st.selectbox(
            "Browse by category",
            [""] + browse_categories,
            key="browse_cat",
        )

    with br2:
        if browse_cat:
            # Filter products in this category and build a readable label
            cat_rows = df[df["Product Category"] == browse_cat].copy()
            cat_rows["_label"] = (
                cat_rows["Product Description"].fillna("").astype(str).str[:80]
                + "  |  "
                + cat_rows["Brand"].fillna("").astype(str)
                + "  |  Row " + (cat_rows.index + 1).astype(str)
            )
            product_options = cat_rows["_label"].tolist()
            selected_product = st.selectbox(
                f"Select a product ({len(cat_rows):,} records)",
                [""] + product_options,
                key="browse_product",
            )
        else:
            selected_product = ""
            st.selectbox("Select a product", ["Pick a category first"], disabled=True)

    # Buttons: load selected product or clear fields
    load_col, random_col, clear_col = st.columns(3)

    with load_col:
        load_clicked = st.button("Load Selected Product", width="stretch")

    with random_col:
        random_clicked = st.button("Load Random Example", width="stretch")

    with clear_col:
        clear_clicked = st.button("Clear Fields", width="stretch")

    # Helper to fill session_state from a DataFrame row
    def _fill_from_row(sample):
        sample = sample.fillna("")
        st.session_state["ex_product_desc"] = str(sample.get("Product Description", "") or "")
        st.session_state["ex_incident_desc"] = str(sample.get("Incident Description", "") or "")
        st.session_state["ex_brand"] = str(sample.get("Brand", "") or "")
        st.session_state["ex_manufacturer"] = str(sample.get("Manufacturer / Importer / Private Labeler Name", "") or "")
        st.session_state["ex_model_num"] = str(sample.get("Model Name or Number", "") or "")
        st.session_state["ex_category"] = str(sample.get("Product Category", "") or "")
        st.session_state["ex_subcategory"] = str(sample.get("Product Sub Category", "") or "")
        st.session_state["ex_type"] = str(sample.get("Product Type", "") or "")

    if load_clicked:
        if browse_cat and selected_product:
            _fill_from_row(cat_rows.loc[cat_rows["_label"] == selected_product].iloc[0])
            st.rerun()
        else:
            st.warning("Choose a category and a product to load a selected example.")

    if random_clicked:
        _fill_from_row(df.sample(1).iloc[0])
        st.rerun()

    if clear_clicked:
        for fk in field_keys:
            st.session_state[fk] = ""
        st.rerun()

    st.markdown("---")

    # Get the unique values for the dropdown menus
    categories = sorted(df["Product Category"].dropna().unique().tolist())
    subcategories = sorted(df["Product Sub Category"].dropna().unique().tolist())
    product_types = sorted(df["Product Type"].dropna().unique().tolist())

    # Helper to find the index of a value in a list, or 0 if not found
    def _idx(lst, val):
        """Find index of val in [''] + lst, return 0 if not found."""
        full = [""] + lst
        try:
            return full.index(val)
        except ValueError:
            return 0

    # Build the input form — fields are pre-filled from session_state
    # but remain fully editable so the user can modify them.
    with st.form("risk_form"):

        # Row 1: text areas for descriptions
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            product_desc = st.text_area(
                "Product Description",
                value=st.session_state["ex_product_desc"],
                placeholder="e.g. Samsung top load washing machine, white",
                height=100,
            )
        with r1c2:
            incident_desc = st.text_area(
                "Incident / Complaint Text",
                value=st.session_state["ex_incident_desc"],
                placeholder="e.g. The washing machine exploded during spin cycle...",
                height=100,
            )

        # Row 2: brand, manufacturer, model
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            brand = st.text_input(
                "Brand",
                value=st.session_state["ex_brand"],
                placeholder="e.g. Samsung",
            )
        with r2c2:
            manufacturer = st.text_input(
                "Manufacturer / Importer / Private Labeler",
                value=st.session_state["ex_manufacturer"],
                placeholder="e.g. Samsung Electronics America",
            )
        with r2c3:
            model_num = st.text_input(
                "Model Name or Number",
                value=st.session_state["ex_model_num"],
                placeholder="e.g. WA50F9A8DSW",
            )

        # Row 3: category dropdowns (pre-selected from session_state)
        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            selected_category = st.selectbox(
                "Product Category",
                [""] + categories,
                index=_idx(categories, st.session_state["ex_category"]),
            )
        with r3c2:
            selected_subcategory = st.selectbox(
                "Product Sub Category",
                [""] + subcategories,
                index=_idx(subcategories, st.session_state["ex_subcategory"]),
            )
        with r3c3:
            selected_type = st.selectbox(
                "Product Type",
                [""] + product_types,
                index=_idx(product_types, st.session_state["ex_type"]),
            )

        # Submit button
        submitted = st.form_submit_button(
            "Calculate Review Priority",
            width="stretch",
            type="primary",
        )

    # When the form is submitted, compute and display the risk score
    if submitted:

        try:
            input_row = prepare_prediction({
                "Product Description": product_desc,
                "Incident Description": incident_desc,
                "Manufacturer / Importer / Private Labeler Name": manufacturer,
                "Brand": brand,
                "Model Name or Number": model_num,
                "Product Category": selected_category,
                "Product Sub Category": selected_subcategory,
                "Product Type": selected_type,
            })
        except ValueError as error:
            st.warning(str(error))
        else:
            try:
                scorer = cached_scorer(df)["model"]
                association = scorer.predict_proba(input_row)[0, 1]
            except ValueError as error:
                st.error(f"Could not score this report. {error}")
                st.stop()

            # Compute the hazard severity signal from the text
            full_text = (product_desc or "") + " " + (incident_desc or "")
            sev_score, crit_hits, high_hits, mod_hits = compute_severity_score(full_text)

            # Blend historical association and severity into a review priority score
            final_score = composite_risk_score(association, sev_score)

            # Classify into Low / Moderate / High based on composite score
            if final_score < 0.25:
                bucket = "Low"
                css_class = "risk-low"
                action = "Lower priority under this scoring rule. Review urgent hazards independently."
            elif final_score < 0.55:
                bucket = "Moderate"
                css_class = "risk-moderate"
                action = (
                    "Review product reports. Consider proactive quality "
                    "checks and gather more incident data."
                )
            else:
                bucket = "High"
                css_class = "risk-high"
                action = (
                    "Escalate for detailed safety review. Gather related "
                    "incident data and consider contacting the manufacturer."
                )

            # Display the big composite score box
            st.markdown("")
            rc1, rc2, rc3 = st.columns([1, 2, 1])
            with rc2:
                st.markdown(
                    f'<div class="risk-box {css_class}">'
                    f'<h2>{final_score:.2f} / 1</h2>'
                    f'<p>Combined Review Priority Score &middot; '
                    f'<strong>{bucket}</strong></p></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")

            # Show the two individual signals side by side
            sig1, sig2 = st.columns(2)
            with sig1:
                st.metric(
                    "Historical Recall Association",
                    f"{association:.2f}",
                    help=(
                        "Logistic regression score for association with a historical "
                        "recall-linkage label in the capstone sample."
                    ),
                )
            with sig2:
                st.metric(
                    "Incident Severity Signal",
                    f"{sev_score:.2f}",
                    help=(
                        "Rule-based score from the severity of language "
                        "in the product and incident descriptions (critical, high, "
                        "moderate hazard terms)"
                    ),
                )

            st.markdown("")
            st.markdown(f"**Suggested Action:** {action}")

            # Expandable breakdown of what influenced the score
            with st.expander("What influenced this score?"):

                # Show the math behind the composite score
                st.markdown("**Composite Score Breakdown**")
                sev_weight = 0.30 + 0.30 * sev_score
                model_weight = 1.0 - sev_weight
                st.markdown(
                    f"- ML model contribution: {association:.2f} "
                    f"x {model_weight:.0%} weight = "
                    f"{association * model_weight:.2f}"
                )
                st.markdown(
                    f"- Severity contribution: {sev_score:.2f} "
                    f"x {sev_weight:.0%} weight = "
                    f"{sev_score * sev_weight:.2f}"
                )

                # Show which hazard words were detected
                st.markdown("")
                st.markdown("**Hazard Language Detected**")

                tokens = hazard_tokens(full_text)
                if crit_hits > 0:
                    crit_found = tokens & SEVERITY_CRITICAL
                    st.markdown(
                        f"- Critical terms ({crit_hits}): "
                        + ", ".join(sorted(crit_found)[:6])
                    )
                if high_hits > 0:
                    high_found = tokens & SEVERITY_HIGH
                    st.markdown(
                        f"- High-severity terms ({high_hits}): "
                        + ", ".join(sorted(high_found)[:6])
                    )
                if mod_hits > 0:
                    mod_found = tokens & SEVERITY_MODERATE
                    st.markdown(
                        f"- Moderate terms ({mod_hits}): "
                        + ", ".join(sorted(mod_found)[:6])
                    )
                if crit_hits == 0 and high_hits == 0 and mod_hits == 0:
                    st.markdown("- No hazard-related language detected")

            with st.expander("How to interpret the signals"):
                st.markdown("""
- **Historical Recall Association** summarizes similarities to reports linked
  to recalls in the capstone sample. Sampling and imperfect record linkage
  affect this score.
- **Incident Severity Signal** counts hazard terms using fixed weights.
  It can miss context, including negation, and cannot confirm what happened.
- **Combined Review Priority Score** gives more weight to the severity
  signal when it finds more severe language. Use it as an exploratory
  triage aid. Official CPSC safety assessments require independent evidence.
                """)


# PAGE 4: CATEGORY INSIGHTS
# For each product category, shows:
# - Row count and recall rate
# - Model performance within that category
# - Top keywords pushing toward/away from recall


# PAGE: MODEL DETAILS (was Dashboard)
# Shows the best model's metrics, confusion matrix, and a comparison
# table and bar chart of all 9 models ranked by F1 score.

elif page == "Model Details":

    st.header("Model Details")
    st.markdown("Historical capstone F1 was **0.688** for the general Linear SVM and "
                "**0.784** for Home Maintenance and Structures.")
    st.caption("The bundled data and original random holdout informed prior capstone "
               "development. Both holdouts below are retrospective. Candidate and threshold "
               "selection for this run uses development folds only.")
    report_path = DATA_PATH.with_name("evaluation_results.json")
    try:
        report = cached_benchmark(
            DATA_PATH, report_path, DATA_PATH.stat().st_mtime_ns, report_path.stat().st_mtime_ns,
        )
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Benchmark report unavailable. {error}")
        st.code("python app/evaluation.py > data/evaluation_results.json", language="bash")
        st.stop()

    random = report["random"]
    display_name = random["selection"]["model"]
    random_test = random["retrospective_holdout"]
    m = random_test["metrics"]
    comparison = []
    for name, key in (("Original random", "random"), ("Related-report grouped", "grouped")):
        result = report[key]
        selection = result["selection"]
        holdout = result["retrospective_holdout"]
        comparison.append({
            "Split": name,
            "Selected model": selection["model"],
            "Development CV F1": result["development_cv"][selection["model"]]["f1"],
            "Tuned cutoff": selection["threshold"],
            "Holdout rows": holdout["rows"],
            "Holdout positives": holdout["positives"],
            "Holdout F1": holdout["metrics"]["f1"],
        })
    comparison = pd.DataFrame(comparison)
    st.dataframe(comparison, hide_index=True)
    st.caption("An empty tuned cutoff indicates the classifier’s default decision rule. "
               "Development CV scores were used to select candidates and can be optimistic.")

    # Section label
    st.markdown(
        '<div class="section-header">Original Random Split · Retrospective Holdout</div>',
        unsafe_allow_html=True,
    )

    # Display 5 metric cards in a row
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Selected Model", display_name)
    c2.metric("Accuracy", f"{m['accuracy']:.4f}")
    c3.metric("Precision", f"{m['precision']:.4f}")
    c4.metric("Recall", f"{m['recall']:.4f}")
    c5.metric("F1 Score", f"{m['f1']:.4f}")

    st.markdown("")

    # Two columns: confusion matrix on left, comparison table on right
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown(
            '<div class="section-header">Confusion Matrix</div>',
            unsafe_allow_html=True,
        )

        # Build a Plotly heatmap for the confusion matrix
        cm = np.asarray(random_test["confusion_matrix"])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm[::-1],
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 1", "Actual 0"],
            text=cm[::-1],
            texttemplate="%{text}",
            textfont=dict(size=18, family="Space Mono"),
            colorscale=[[0, "#1e293b"], [1, "#3b82f6"]],
            showscale=False,
            hoverinfo="skip",
        ))
        fig_cm.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans"),
            xaxis=dict(side="bottom"),
        )
        st.plotly_chart(fig_cm, width="stretch")

    with col_right:
        st.markdown(
            '<div class="section-header">Random Split · Development Cross-Validation</div>',
            unsafe_allow_html=True,
        )

        # Build a DataFrame from all model results and sort by F1
        results_table = pd.DataFrame(random["development_cv"]).T
        results_table.index.name = "Model"
        results_table = results_table.sort_values("f1", ascending=False)

        # Display as a styled table with the best values highlighted
        st.dataframe(
            results_table.style
                .format("{:.4f}")
                .highlight_max(axis=0, props="color: #3b82f6; font-weight: bold;"),
            width="stretch",
            height=380,
        )

    # F1 Score comparison bar chart
    st.markdown("")
    st.markdown(
        '<div class="section-header">Random Split · Development CV F1</div>',
        unsafe_allow_html=True,
    )

    bar_data = results_table.reset_index().rename(columns={"index": "Model"})

    # Highlight the best model in blue, rest in gray
    bar_colors = []
    for name in bar_data["Model"]:
        if name == display_name:
            bar_colors.append("#3b82f6")
        else:
            bar_colors.append("#334155")

    fig_f1 = go.Figure(go.Bar(
        x=bar_data["f1"],
        y=bar_data["Model"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.4f}" for v in bar_data["f1"]],
        textposition="outside",
        textfont=dict(family="Space Mono", size=11),
    ))
    fig_f1.update_layout(
        height=max(320, len(bar_data) * 38),
        margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor="#1e293b",
            range=[0, max(bar_data["f1"]) * 1.15],
        ),
        yaxis=dict(autorange="reversed"),
        font=dict(family="DM Sans", size=12),
    )
    st.plotly_chart(fig_f1, width="stretch")

    # Expandable explanation of the metrics
    with st.expander("How to interpret these metrics"):
        st.markdown("""
**Accuracy** is the fraction of correct predictions. The unlinked class
accounts for most of this sample.

**Precision** is the share of predicted positive reports with a positive
linkage label.

**Recall (Sensitivity)** is the share of positive linkage labels that the
classifier recovered.

**F1 Score** is the harmonic mean of precision and recall. The primary metric
for this project because both false alarms and missed recalls matter.

**Confusion Matrix** shows actual labels on rows and predicted labels on columns. Off-diagonal cells are errors.

**SMOTE Models** : Synthetic Minority Over-sampling Technique generates
synthetic examples of the minority class (recalled) to address class
imbalance.

**Threshold Tuning** : Instead of using the default 0.50 cutoff for
Logistic Regression, the tuned version picks the threshold that
maximizes F1 on out-of-fold cross-validation predictions.

**XGBoost** : Gradient-boosted decision tree ensemble with
balanced row weights calculated on each training fold.

**Random Forest** : Bagged decision tree ensemble with balanced
class weights.

**SGD Classifier** : Stochastic Gradient Descent with log loss
(equivalent to logistic regression optimized via SGD), using balanced
class weights.
        """)


# PAGE 2: DATASET INSIGHTS
# Shows class balance, recall rate by category, and a data preview.

