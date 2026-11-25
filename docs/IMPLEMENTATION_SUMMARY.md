# Implementation summary

## Implemented and verified

- Preserved the original license, tracked labeled dataset, slide deck, screenshot PDF and Git history. Archived the original README and documented the source audit and limitations.
- Added CPSC CSV ingestion and schema checks, bounded token-based record linkage, dated acceptance criteria, auditable scoring, reason codes and reconstructed nullable proxy labels. The local export run read 65,058 incidents and 9,666 recall listings, retained 9,623 recall IDs, and scored 1,144,551 candidate pairs. With model and UPC/GTIN evidence, 146 reports met the unique high-confidence rule. The older-audit recovery check found 204 of 207 date-eligible candidates, compared with 170 in the text-only version.
- Added a sparse supervised training CLI with a dummy baseline, logistic regression, Linear SVM, Complement NB and XGBoost. It compares class weights and random oversampling, keeps training, validation, calibration and test separate, and exports full metrics and trusted model bundles.
- Ran full training on the 3,780-row historical CSV across all three split designs. Validation-selected holdout F1 was 0.562 random, 0.402 product-grouped, and 0.019 chronological with group purging. Corresponding AP was 0.551, 0.412, and 0.144. The historical general F1 of 0.688 and category F1 of 0.784 remain historical.
- Verified the Streamlit navigation, metrics and scoring form in Chrome, and recorded actual dashboard screenshots. Verified FastAPI validation, unavailable-model behavior, and successful model-backed inference. Built the Docker image, trained a lightweight model inside it, and checked both service health endpoints plus model info and API prediction as the image's non-root user for the original portfolio release. The identifier-aware follow-up was verified through the CLI and local tests.
- Verified formatting, lint and dependency compatibility. Pytest passed all 11 tests on Python 3.11 and 3.13 with no failures. FastAPI’s TestClient emitted a Starlette HTTP client deprecation warning. Dependency audits of runtime and development declarations returned no known vulnerabilities for the resolved packages at execution. A narrow text signature scan found none of the specified credential patterns in Git history.
- Documented the capstone context, architecture, local Docker and API commands, practical limitations, interview answers and resume wording.

## Implemented but not fully verified

- GitHub Actions defines Python 3.11 and 3.13 jobs for compatibility, Ruff and pytest. The commands were checked locally. A hosted Actions run requires publishing the branch.
- The reconstructed linker ran on the available raw snapshots, but its high-confidence rules lack independent manual precision and recall estimates. Candidate outputs and contact-shaped strings in legacy narratives need human review before public distribution.
- The Docker image ran locally on arm64. Other architectures and hosted service configurations have not been tested here.

## Not implemented

- Validated prospective early-warning prediction. The project lacks adjudicated longitudinal outcomes and point-in-time report snapshots, and the chronological proxy benchmark shows weak transfer.
- Independent reproduction of the original category-trained experiment and exact original linkage builder. The source history does not contain the builder, original experiment memberships or original fitted artifacts. New category breakdowns slice frozen predictions.
- Public deployment, experiment service integration, full data de-identification, and the optional MLflow integration.

## Run locally

Use [Quick Start](../README.md#quick-start) for installation and training, [Data Pipeline](DATA_PIPELINE.md) for the raw exports, [Model Evaluation](MODEL_EVALUATION.md) for metrics, and [Deployment](DEPLOYMENT.md) for Docker and the API. The benchmark JSON under `results/metrics` includes the dataset fingerprint, run settings and dependency versions. Local raw inputs, reconstruction outputs and trained bundles are excluded from Git.

## Publication checklist

- Verify the scope of the local feature branch and preserve the archived research and MIT attribution.
- Review the existing dataset, screenshots and derived artifacts for personal information and redistribution restrictions. Keep model bundles, candidate tables, raw exports and images built from the unreviewed CSV private.
- Push the branch only after the repository owner approves. Observe the hosted CI run after publication.
