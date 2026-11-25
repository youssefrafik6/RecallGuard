"""FastAPI service with operator-configured model loading."""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from recallguard.inference import (
    LIMITATION,
    IncidentRequest,
    Prediction,
    configured_model,
    predict,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = configured_model()
    yield


app = FastAPI(title="RecallGuard", version="0.1.0", lifespan=lifespan)


class Health(BaseModel):
    status: Literal["ok"]
    model_available: bool


class ModelInfo(BaseModel):
    model: str
    design: str
    features: list[str]
    limitation: str


def available_model(request: Request) -> dict:
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(503, "No compatible trained model is configured")
    return bundle


@app.get("/health", response_model=Health)
def health(request: Request):
    return Health(status="ok", model_available=request.app.state.bundle is not None)


@app.get("/model-info", response_model=ModelInfo)
def model_info(request: Request):
    metadata = available_model(request)["metadata"]
    return ModelInfo(
        model=metadata["selected"],
        design=metadata["design"],
        features=metadata["features"],
        limitation=LIMITATION,
    )


@app.post("/predict", response_model=Prediction)
def predict_route(body: IncidentRequest, request: Request):
    return predict(available_model(request), body)
