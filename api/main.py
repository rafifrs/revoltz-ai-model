import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schema import (
    Model1InfoResponse,
    ModelInfoResponse,
    PredictPackRequest,
    PredictPackResponse,
    PredictRequest,
    PredictResponse,
    PredictSummary,
)
from src.predictor import get_model, get_model1, predict_cells_with_recommendations, predict_pack_health


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_model()
    try:
        get_model1()
    except FileNotFoundError:
        pass
    yield


app = FastAPI(
    title="ReVoltz Cell Matcher API",
    description="Battery cell compatibility engine for repack candidate screening.",
    version="1.0.0",
    lifespan=lifespan,
)

allowed_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]
allow_credentials = not (len(allowed_origins) == 1 and allowed_origins[0] == "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "ReVoltz Cell Matcher"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    model = get_model()
    metadata = model.metadata
    return ModelInfoResponse(
        model_version=metadata.get("model_version", "unknown"),
        selected_clusters=metadata.get("selected_clusters", 0),
        snapshot_cycle=metadata.get("snapshot_cycle"),
        training_samples=metadata.get("training_samples", 0),
        non_outlier_training_samples=metadata.get("non_outlier_training_samples", 0),
        compatibility_thresholds=metadata.get("compatibility_thresholds", {}),
        iqr_bounds=metadata.get("iqr_bounds", {}),
        silhouette_score=metadata.get("silhouette_score"),
    )


@app.get("/model1-info", response_model=Model1InfoResponse)
def model1_info() -> Model1InfoResponse:
    model = get_model1()
    metadata = model.metadata
    return Model1InfoResponse(
        model_version=metadata.get("model_version", "unknown"),
        training_rows=metadata.get("training_rows", 0),
        eval_rows=metadata.get("eval_rows", 0),
        mae=metadata.get("mae", 0.0),
        rmse=metadata.get("rmse", 0.0),
        r2=metadata.get("r2", 0.0),
    )


@app.post("/predict-cells", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    cells_input = [cell.model_dump() for cell in request.cells]
    assembly_request = request.assembly_request
    results, recommended_packs = predict_cells_with_recommendations(
        cells_input,
        target_pack_size=assembly_request.target_pack_size if assembly_request else None,
        max_packs=assembly_request.max_packs if assembly_request else None,
        allow_partial_packs=assembly_request.allow_partial_packs if assembly_request else True,
    )

    compatible_pairs = set()
    for result in results:
        for peer in result["compatible_with"]:
            compatible_pairs.add(tuple(sorted((result["cell_id"], peer))))

    status_breakdown: dict[str, int] = {}
    for result in results:
        status_breakdown[result["status"]] = status_breakdown.get(result["status"], 0) + 1

    n_outlier = sum(1 for result in results if result["is_outlier"])
    n_valid = len(results) - n_outlier
    cluster_count = len({result["cluster"] for result in results if result["cluster"] is not None})

    return PredictResponse(
        results=results,
        summary=PredictSummary(
            total_cells=len(results),
            outlier_count=n_outlier,
            valid_for_repack=n_valid,
            clusters_found=cluster_count,
            compatible_pairs=len(compatible_pairs),
            recommended_pack_count=len(recommended_packs),
            status_breakdown=status_breakdown,
        ),
        recommended_packs=recommended_packs,
    )


@app.post("/predict-pack", response_model=PredictPackResponse)
def predict_pack(request: PredictPackRequest) -> PredictPackResponse:
    packs_input = [pack.model_dump() for pack in request.packs]
    results = predict_pack_health(packs_input)
    action_counts: dict[str, int] = {}
    for row in results:
        action = row["recommended_action"]
        action_counts[action] = action_counts.get(action, 0) + 1
    avg_soh = sum(row["predicted_soh"] for row in results) / len(results)
    avg_conf = sum(row["confidence_score"] for row in results) / len(results)
    return PredictPackResponse(
        results=results,
        summary={
            "total_packs": len(results),
            "average_predicted_soh": avg_soh,
            "average_confidence_score": avg_conf,
            "recondition_count": action_counts.get("recondition", 0),
            "repack_count": action_counts.get("repack", 0),
            "recycle_count": action_counts.get("recycle", 0),
        },
    )
