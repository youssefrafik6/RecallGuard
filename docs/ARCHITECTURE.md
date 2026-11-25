# Architecture

The [README diagram](../README.md#architecture) shows two data paths. The reconstruction path accepts raw incident and recall CSVs and produces auditable candidate and nullable observation outputs. The training path uses the historical labeled CSV. Ingestion restricts accepted product and report fields and resolves ambiguous repeated identifiers before either path.

The `recallguard` package centralizes ingestion, linkage, split construction, metrics, model training and shared inference. The CLI manages local paths and writes atomic model bundles plus JSON. Learned text and structured transformations fit only on training rows. The training command selects variants using validation, calibrates separately, and scores the frozen choices against the final holdout.

Streamlit caches local data and model resources. The dashboard reads training and performance metadata and uses the same request model and prediction function as FastAPI. The API loads the artifact at startup and reports 503 on artifact-dependent endpoints when the bundle is unavailable. Requests supply product fields only. Trusted artifact paths are controlled by the operator.
