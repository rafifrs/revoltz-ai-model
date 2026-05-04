from __future__ import annotations

from pathlib import Path

from src.config import DEFAULT_MODEL1_PATH, DEFAULT_MODEL_PATH
from src.model.model1_soh import SoHPredictor
from src.model.model2_cell_matcher import CellMatcher
from src.pack_recommender import recommend_packs


_model: CellMatcher | None = None
_model1: SoHPredictor | None = None


def get_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> CellMatcher:
    global _model
    if _model is None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {path}. Run `python train.py` first."
            )
        _model = CellMatcher.load(path)
    return _model


def predict_cells(cells: list[dict]) -> list[dict]:
    return get_model().predict(cells)


def get_model1(model_path: str | Path = DEFAULT_MODEL1_PATH) -> SoHPredictor:
    global _model1
    if _model1 is None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model 1 artifact not found at {path}. Run `python train_model1.py` first.")
        _model1 = SoHPredictor.load(path)
    return _model1


def predict_pack_health(packs: list[dict]) -> list[dict]:
    return get_model1().predict(packs)


def predict_cells_with_recommendations(
    cells: list[dict],
    target_pack_size: int | None = None,
    max_packs: int | None = None,
    allow_partial_packs: bool = True,
) -> tuple[list[dict], list[dict]]:
    model = get_model()
    results = model.predict(cells)
    recommended_packs: list[dict] = []

    if target_pack_size is not None:
        recommended_packs = recommend_packs(
            results=results,
            target_pack_size=target_pack_size,
            ir_threshold=model.compatibility_ir_threshold,
            ocv_threshold=model.compatibility_ocv_threshold,
            max_packs=max_packs,
            allow_partial_packs=allow_partial_packs,
        )

    return results, recommended_packs
