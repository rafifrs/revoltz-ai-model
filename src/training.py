from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, DATA_DIR, DEFAULT_MODEL_PATH, DEFAULT_SNAPSHOT_CYCLE
from src.data_loader import load_all_cells
from src.feature_extractor import extract_snapshot_features
from src.labeler import generate_compatibility_pairs
from src.model.model2_cell_matcher import CellMatcher


def build_training_dataframe(cells: list[dict], snapshot_cycle: int) -> pd.DataFrame:
    snapshots: list[dict] = []
    for cell in cells:
        snapshot = extract_snapshot_features(cell, at_cycle=snapshot_cycle)
        if snapshot:
            snapshots.append(snapshot)

    df = pd.DataFrame(snapshots)
    if df.empty:
        raise ValueError("No snapshots could be extracted from the dataset.")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ReVoltz cell matcher model.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Directory containing HNEI PKL files.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Output path for the trained model.")
    parser.add_argument("--snapshot-cycle", type=int, default=DEFAULT_SNAPSHOT_CYCLE, help="Cycle index used as the training snapshot.")
    parser.add_argument("--clusters", type=int, default=None, help="Optional fixed number of KMeans clusters.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cells = load_all_cells(args.data_dir)
    df = build_training_dataframe(cells, snapshot_cycle=args.snapshot_cycle)
    valid_df = df.dropna(subset=["IR_ohm", "OCV_V"]).copy()
    if valid_df.empty:
        raise ValueError("No valid snapshots with both IR and OCV were extracted.")

    model = CellMatcher(n_clusters=args.clusters)
    model.fit(valid_df, snapshot_cycle=args.snapshot_cycle)
    model.save(args.model_path)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)

    snapshots_path = ARTIFACTS_DIR / "snapshot_features.csv"
    valid_snapshots_path = ARTIFACTS_DIR / "snapshot_features_valid.csv"
    pairs_path = ARTIFACTS_DIR / "compatibility_pairs.csv"
    metrics_path = ARTIFACTS_DIR / "training_metadata.json"

    df.to_csv(snapshots_path, index=False)
    valid_df.to_csv(valid_snapshots_path, index=False)
    generate_compatibility_pairs(valid_df).to_csv(pairs_path, index=False)
    metrics_path.write_text(json.dumps(model.metadata, indent=2))

    print(f"Loaded cells: {len(cells)}")
    print(f"Snapshot rows: {len(df)}")
    print(f"Valid snapshot rows: {len(valid_df)}")
    print(f"Model saved to: {args.model_path}")
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")
    print(json.dumps(model.metadata, indent=2))
    return 0

