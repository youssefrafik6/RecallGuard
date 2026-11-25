# Local demo and deployment

Install the project and run Streamlit or FastAPI using the [Quick Start](../README.md#quick-start). Supply a trusted bundle created with the same model dependency versions. The app can start in exploration mode without an artifact. Runtime requests never accept artifact paths or uploads.

## API behavior

- `GET /health` returns HTTP 200 with `{"status":"ok","model_available":false}` when the service is live and no model is configured.
- `GET /model-info` returns the model family, evaluation design, feature schema and interpretation limit.
- `POST /predict` accepts a JSON object with `product`, `description`, `category`, `brand`, `manufacturer` and `product_type`. Supply at least one description. Description fields allow 16,000 characters each. Short fields allow 240 characters. Unknown fields and invalid bodies return 422. Model-dependent routes return 503 without a compatible bundle.
- A successful prediction contains `model`, `priority_score`, `priority`, `contributions` and `limitation`. Individual contributions are available for linear and XGBoost models. See `/docs` for the live schema and [the README](../README.md#api) for a curl request.

For the README's example request, the full grouped benchmark returned the following abridged response. The live endpoint returns all contributions.

```json
{
  "model": "logistic",
  "priority_score": 0.10456518644135887,
  "priority": "Low",
  "contributions": [{"feature": "category__category_kitchen", "effect": -0.2585993999785451}],
  "limitation": "Priority reflects association with historical linkage labels in a sampled dataset."
}
```

## Docker

```bash
docker build -t recallguard:local .

# Browser demo without a model
docker run --rm -p 127.0.0.1:8501:8501 recallguard:local

# Train inside the image so artifact dependencies match the serving image.
# This one-off step runs as the host UID to write into your local artifact directory.
mkdir -p results/runs/container
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/results/runs/container:/runs" recallguard:local \
  recallguard train --design group --out /runs

# The serving UID needs read access to your trusted model.
chmod a+r results/runs/container/group.joblib
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$PWD/results/runs/container:/models:ro" \
  -e RECALLGUARD_MODEL=/models/group.joblib recallguard:local \
  uvicorn recallguard.api:app --host 0.0.0.0
```

For model-backed Streamlit, use the same mount and environment with port 8501 and the image's default command. The image runs as a non-root user by default. Public hosting is a separate step. Review redistribution and narrative exposure, generate a trusted artifact in the serving environment, and configure TLS, request size, concurrency and access controls before exposing the service. There is no public deployment.

Keep locally built images private until the historical CSV embedded in the image has had a redistribution review. The checked-in sample predates the new data handling and contains contact-shaped strings in free text.
