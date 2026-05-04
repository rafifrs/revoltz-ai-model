from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODEL1_RECONDITION_MIN_SOH, MODEL1_REPACK_MIN_SOH, MODEL1_VERSION
from src.model.common import load_pickle, save_pickle


def _pick_regressor() -> object:
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
        )
    except Exception:
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42)
        except Exception:
            return GradientBoostingRegressor(random_state=42)


@dataclass
class SoHPredictor:
    recondition_min_soh: float = MODEL1_RECONDITION_MIN_SOH
    repack_min_soh: float = MODEL1_REPACK_MIN_SOH
    pipeline: Pipeline | None = None
    residual_quantile: float | None = None
    metadata: dict = field(default_factory=dict)

    numeric_features: tuple[str, ...] = ("ocv_v", "capacity_ah", "cycle_count", "temperature_c", "age_days")
    categorical_features: tuple[str, ...] = ("chemistry",)

    def _build_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("scaler", StandardScaler())]), list(self.numeric_features)),
                ("cat", OneHotEncoder(handle_unknown="ignore"), list(self.categorical_features)),
            ],
            remainder="drop",
        )
        model = _pick_regressor()
        return Pipeline([("preprocessor", preprocessor), ("regressor", model)])

    def _prepare(self, df: pd.DataFrame, with_target: bool = True) -> tuple[pd.DataFrame, pd.Series | None]:
        required = set(self.numeric_features + self.categorical_features)
        if with_target:
            required.add("soh")
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for SoH model: {sorted(missing)}")

        frame = df.copy()
        for col in self.numeric_features:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["chemistry"] = frame["chemistry"].fillna("unknown").astype(str)
        feature_cols = list(self.numeric_features + self.categorical_features)
        frame = frame.dropna(subset=["ocv_v", "capacity_ah", "cycle_count", "temperature_c"] + (["soh"] if with_target else []))
        y = frame["soh"].astype(float) if with_target else None
        return frame[feature_cols], y

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "SoHPredictor":
        x_train, y_train = self._prepare(train_df, with_target=True)
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(x_train, y_train)

        eval_df = val_df if val_df is not None else train_df
        x_eval, y_eval = self._prepare(eval_df, with_target=True)
        preds = self.pipeline.predict(x_eval)
        abs_err = np.abs(y_eval.to_numpy() - preds)
        self.residual_quantile = float(np.quantile(abs_err, 0.9))
        rmse = float(np.sqrt(mean_squared_error(y_eval, preds)))
        self.metadata = {
            "model_version": MODEL1_VERSION,
            "training_rows": int(len(x_train)),
            "eval_rows": int(len(x_eval)),
            "mae": float(mean_absolute_error(y_eval, preds)),
            "rmse": rmse,
            "r2": float(r2_score(y_eval, preds)),
            "residual_q90": self.residual_quantile,
            "triage_thresholds": {
                "recondition_min_soh": self.recondition_min_soh,
                "repack_min_soh": self.repack_min_soh,
            },
        }
        return self

    def _action_from_soh(self, soh: float) -> str:
        if soh >= self.recondition_min_soh:
            return "recondition"
        if soh >= self.repack_min_soh:
            return "repack"
        return "recycle"

    def predict(self, packs: list[dict]) -> list[dict]:
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted.")
        frame = pd.DataFrame(packs)
        x, _ = self._prepare(frame, with_target=False)
        preds = self.pipeline.predict(x)
        q = self.residual_quantile or 0.1
        results: list[dict] = []
        for raw, soh in zip(x.to_dict(orient="records"), preds):
            predicted = float(np.clip(soh, 0.0, 1.2))
            confidence = float(np.clip(1.0 - min(abs(predicted - 0.8), q) / max(q, 1e-6), 0.0, 1.0))
            results.append(
                {
                    "pack_id": raw.get("pack_id"),
                    "predicted_soh": predicted,
                    "confidence_score": confidence,
                    "recommended_action": self._action_from_soh(predicted),
                    "notes": ["Prediction generated from pack-level diagnostic features."],
                }
            )
        return results

    def save(self, path: str | Path) -> None:
        save_pickle(self, path)

    @staticmethod
    def load(path: str | Path) -> "SoHPredictor":
        return load_pickle(path)  # type: ignore[return-value]
