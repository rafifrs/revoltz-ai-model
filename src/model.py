from __future__ import annotations

import multiprocessing
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(multiprocessing.cpu_count()))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import (
    COMPATIBILITY_IR_THRESHOLD,
    COMPATIBILITY_OCV_THRESHOLD,
    IQR_MULTIPLIER,
    MODEL_VERSION,
)


@dataclass
class CellMatcher:
    n_clusters: int | None = None
    cluster_range: tuple[int, int] = (2, 6)
    iqr_multiplier: float = IQR_MULTIPLIER
    compatibility_ir_threshold: float = COMPATIBILITY_IR_THRESHOLD
    compatibility_ocv_threshold: float = COMPATIBILITY_OCV_THRESHOLD
    scaler: StandardScaler = field(default_factory=StandardScaler)
    kmeans: KMeans | None = None
    ir_bounds: tuple[float, float] | None = None
    ocv_bounds: tuple[float, float] | None = None
    metadata: dict = field(default_factory=dict)

    def _compute_iqr_bounds(self, values: np.ndarray) -> tuple[float, float]:
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr
        return float(lower), float(upper)

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"cell_id", "IR_ohm", "OCV_V"}
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        clean_df = df.dropna(subset=["IR_ohm", "OCV_V"]).copy()
        if clean_df.empty:
            raise ValueError("No valid training rows after dropping missing IR/OCV values.")
        return clean_df

    def _select_cluster_count(self, features_scaled: np.ndarray) -> tuple[int, float | None]:
        n_samples = len(features_scaled)
        if n_samples == 1:
            return 1, None

        if self.n_clusters is not None:
            selected = max(1, min(self.n_clusters, n_samples))
            if selected == 1:
                return 1, None
            if selected >= n_samples:
                selected = n_samples - 1
            score = silhouette_score(features_scaled, KMeans(n_clusters=selected, random_state=42, n_init=10).fit_predict(features_scaled))
            return selected, float(score)

        min_k, max_k = self.cluster_range
        candidates = range(max(2, min_k), min(max_k, n_samples - 1) + 1)
        best_k = 1
        best_score: float | None = None

        for k in candidates:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(features_scaled)
            score = silhouette_score(features_scaled, labels)
            if best_score is None or score > best_score:
                best_k = k
                best_score = float(score)

        if best_k == 1:
            return 1, None
        return best_k, best_score

    def fit(self, df: pd.DataFrame, snapshot_cycle: int | None = None) -> "CellMatcher":
        train_df = self._validate_dataframe(df)

        self.ir_bounds = self._compute_iqr_bounds(train_df["IR_ohm"].to_numpy())
        self.ocv_bounds = self._compute_iqr_bounds(train_df["OCV_V"].to_numpy())

        mask = train_df["IR_ohm"].between(*self.ir_bounds) & train_df["OCV_V"].between(*self.ocv_bounds)
        df_clean = train_df[mask].copy()
        if df_clean.empty:
            raise ValueError("All samples were flagged as outliers during training.")

        features = df_clean[["IR_ohm", "OCV_V"]].to_numpy()
        features_scaled = self.scaler.fit_transform(features)

        selected_clusters, silhouette = self._select_cluster_count(features_scaled)
        self.kmeans = KMeans(n_clusters=selected_clusters, random_state=42, n_init=10)
        self.kmeans.fit(features_scaled)

        self.metadata = {
            "model_version": MODEL_VERSION,
            "snapshot_cycle": snapshot_cycle,
            "training_samples": int(len(train_df)),
            "non_outlier_training_samples": int(len(df_clean)),
            "selected_clusters": int(selected_clusters),
            "silhouette_score": silhouette,
            "compatibility_thresholds": {
                "IR_ohm": float(self.compatibility_ir_threshold),
                "OCV_V": float(self.compatibility_ocv_threshold),
            },
            "iqr_bounds": {
                "IR_ohm": self.ir_bounds,
                "OCV_V": self.ocv_bounds,
            },
        }
        return self

    def _is_outlier(self, ir_ohm: float | None, ocv_v: float | None) -> bool:
        if ir_ohm is None or ocv_v is None or self.ir_bounds is None or self.ocv_bounds is None:
            return True
        return not (
            self.ir_bounds[0] <= ir_ohm <= self.ir_bounds[1]
            and self.ocv_bounds[0] <= ocv_v <= self.ocv_bounds[1]
        )

    def _cells_are_compatible(self, left: dict, right: dict) -> bool:
        chemistry_ok = True
        if left.get("chemistry") and right.get("chemistry"):
            chemistry_ok = left["chemistry"] == right["chemistry"]

        return (
            left.get("cluster") is not None
            and left.get("cluster") == right.get("cluster")
            and abs(left["IR_ohm"] - right["IR_ohm"]) < self.compatibility_ir_threshold
            and abs(left["OCV_V"] - right["OCV_V"]) < self.compatibility_ocv_threshold
            and chemistry_ok
        )

    def predict(self, cells: list[dict]) -> list[dict]:
        if self.kmeans is None or self.ir_bounds is None or self.ocv_bounds is None:
            raise RuntimeError("Model has not been fitted.")

        results: list[dict] = []
        valid_rows: list[tuple[int, float, float]] = []

        for cell in cells:
            ir_raw = cell.get("IR_ohm")
            ocv_raw = cell.get("OCV_V")
            ir_ohm = float(ir_raw) if ir_raw is not None else None
            ocv_v = float(ocv_raw) if ocv_raw is not None else None
            outlier = self._is_outlier(ir_ohm, ocv_v)

            notes: list[str] = []
            if outlier:
                notes.append("Outside training distribution or missing required measurements.")

            result = {
                "cell_id": cell["cell_id"],
                "IR_ohm": ir_ohm,
                "OCV_V": ocv_v,
                "chemistry": cell.get("chemistry"),
                "source_pack_id": cell.get("source_pack_id"),
                "source_battery_id": cell.get("source_battery_id"),
                "position_in_pack": cell.get("position_in_pack"),
                "is_outlier": outlier,
                "cluster": None,
                "compatible_with": [],
                "recommended_pack_id": None,
                "status": "discard" if outlier else "candidate",
                "notes": notes,
            }
            results.append(result)

            if not outlier and ir_ohm is not None and ocv_v is not None:
                valid_rows.append((len(results) - 1, ir_ohm, ocv_v))

        if valid_rows:
            indices, ir_values, ocv_values = zip(*valid_rows)
            features = np.column_stack([ir_values, ocv_values])
            clusters = self.kmeans.predict(self.scaler.transform(features))
            for idx, cluster in zip(indices, clusters):
                results[idx]["cluster"] = int(cluster)

        for result in results:
            if result["cluster"] is None:
                continue
            peers = []
            for other in results:
                if other["cell_id"] == result["cell_id"]:
                    continue
                if self._cells_are_compatible(result, other):
                    peers.append(other["cell_id"])
            result["compatible_with"] = peers
            if peers:
                result["notes"].append("Matched with at least one peer inside the compatibility thresholds.")
            else:
                result["notes"].append("Inside the training distribution, but no close peer match was found.")

        return results

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: str | Path) -> "CellMatcher":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
