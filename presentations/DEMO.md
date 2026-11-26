# RecallGuard capstone demo

Open the [original capstone slides](RecallGuard_Capstone_Project.pdf) and [dashboard walkthrough](../screenshots/capstone-dashboard.pdf) before presenting. Youssef Rafik, Southern Connecticut State University, Data Science, Spring 2026.

## Before the session

Run from the repository root using the [Python 3.11 setup](../README.md#run-the-app).

```bash
python -m streamlit run app.py
```

Open the local URL Streamlit prints. Submit the example below once to warm the shared scoring model, then return to **Dataset Insights**. Allow about a minute for the first score and keep Streamlit running through the demo. **Model Details** loads the recorded benchmark, so there is no need to rerun the full evaluation during the presentation.

## Five-minute walkthrough

| Time | Show | Say |
| --- | --- | --- |
| 0:00–1:00 | Original slides, then **Dataset Insights** | CPSC incident reports and recall listings lack a common product ID. The capstone linked records using product identifiers and text evidence. The bundled sample contains 3,780 reports, including 420 with positive linkage labels. |
| 1:00–2:00 | **Model Details** | The original capstone reported F1 of 0.688 for the general model and 0.784 for Home Maintenance and Structures. In the revised evaluation, model and cutoff selection use development folds. The held-out random split scored 0.6753 F1 and the grouped split scored 0.3876. |
| 2:00–3:00 | Grouped comparison in **Model Details** | The random split shared 50 recall-candidate identifiers across its boundary, touching 106 holdout reports. Grouping removes that measured crossover. The partitions contain different reports, so the F1 gap describes sensitivity to the split and cannot isolate a single cause. |
| 3:00–4:30 | **Risk Scorer** | Score the fictional example below, then expand **What influenced this score?**. The classifier reflects historical linkage patterns. The severity rules flag language. The combined score prioritizes review on a 0–1 scale. |
| 4:30–5:00 | **Category Insights** or the original slides | Category metrics summarize the scorer on development folds. Both revised holdouts reuse the historical sample, and official safety assessments come from the CPSC. |

### Fictional example for the live demo

- **Product Description** — Countertop electric kettle with a detachable power cord.
- **Incident / Complaint Text** — The cord sparked and the plastic housing caught fire. No injuries were reported.
- Leave the optional fields blank and select **Calculate Review Priority**.

The rehearsal displayed **0.01** historical association, **0.35** incident severity, and **0.15 / 1** combined review priority. The low priority bucket is a result of this scoring rule. A report of fire still warrants independent safety review. Point out that the rules can count a word like “injuries” even when the report says no injuries occurred. This is a concrete limitation of the severity signal.

If the live session is unavailable, use the original slides for the research and the [evaluation report](../EVALUATION.md) for the revised results. The archived dashboard walkthrough shows the original capstone interface.
