from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import ARTIFACTS_DIR, DEFAULT_MODEL1_PATH, MODEL1_DATA_DIR
from src.model.model1_soh import SoHPredictor
from src.model1_data.build_dataset import build_nasa_arc_dataset, split_dataset_by_battery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model 1 SoH Predictor from NASA ARC data.")
    parser.add_argument("--data-dir", default=str(MODEL1_DATA_DIR), help="Directory containing NASA ARC zip files.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL1_PATH), help="Output path for SoH model artifact.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)

    df = build_nasa_arc_dataset(data_dir)
    train_df, val_df, test_df = split_dataset_by_battery(df)

    model = SoHPredictor().fit(train_df, val_df=val_df)
    model.save(model_path)

    test_preds = pd.DataFrame(model.predict(test_df.to_dict(orient="records")))
    eval_df = pd.concat([test_df.reset_index(drop=True), test_preds], axis=1)
    metrics = {
        "mae": float(mean_absolute_error(eval_df["soh"], eval_df["predicted_soh"])),
        "rmse": float(mean_squared_error(eval_df["soh"], eval_df["predicted_soh"]) ** 0.5),
        "r2": float(r2_score(eval_df["soh"], eval_df["predicted_soh"])),
        "rows": int(len(eval_df)),
        "batteries": int(eval_df["battery_id"].nunique()),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_df.head(500).to_csv(ARTIFACTS_DIR / "model1_dataset_snapshot.csv", index=False)
    pd.DataFrame([model.metadata]).to_json(ARTIFACTS_DIR / "model1_training_metadata.json", orient="records", indent=2)
    Path(ARTIFACTS_DIR / "model1_eval_metrics.json").write_text(json.dumps(metrics, indent=2))

    regressor = model.pipeline.named_steps["regressor"] if model.pipeline else None
    if regressor is not None and hasattr(regressor, "feature_importances_"):
        preprocessor = model.pipeline.named_steps["preprocessor"]
        feature_names = list(preprocessor.get_feature_names_out())
        fi_df = pd.DataFrame({"feature": feature_names, "importance": regressor.feature_importances_}).sort_values("importance", ascending=False)
        fi_df.to_csv(ARTIFACTS_DIR / "model1_feature_importance.csv", index=False)
    else:
        pd.DataFrame(columns=["feature", "importance"]).to_csv(ARTIFACTS_DIR / "model1_feature_importance.csv", index=False)

    print(f"Model 1 trained rows: {len(train_df)}")
    print(f"Model 1 saved to: {model_path}")
    print(json.dumps(metrics, indent=2))
    return 0
