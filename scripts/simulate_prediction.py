from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import predict_cells_with_recommendations


DEFAULT_PAYLOAD = {
    "cells": [
        {"cell_id": "demo_cell_A", "source_pack_id": "pack_alpha", "IR_ohm": 0.075, "OCV_V": 3.00},
        {"cell_id": "demo_cell_B", "source_pack_id": "pack_beta", "IR_ohm": 0.077, "OCV_V": 3.02},
        {"cell_id": "demo_cell_C", "source_pack_id": "pack_gamma", "IR_ohm": 0.071, "OCV_V": 3.01},
        {"cell_id": "demo_cell_D", "source_pack_id": "pack_delta", "IR_ohm": 0.120, "OCV_V": 3.40},
    ],
    "assembly_request": {
        "target_pack_size": 3,
        "allow_partial_packs": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate workshop cell matching locally.")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to a JSON file shaped like the /predict-cells request body.",
    )
    return parser.parse_args()


def load_payload(input_path: str | None) -> dict:
    if input_path is None:
        return DEFAULT_PAYLOAD

    path = Path(input_path)
    return json.loads(path.read_text())


def main() -> int:
    args = parse_args()
    payload = load_payload(args.input)
    cells = payload["cells"]
    assembly_request = payload.get("assembly_request")
    results, recommended_packs = predict_cells_with_recommendations(
        cells,
        target_pack_size=assembly_request.get("target_pack_size") if assembly_request else None,
        max_packs=assembly_request.get("max_packs") if assembly_request else None,
        allow_partial_packs=assembly_request.get("allow_partial_packs", True) if assembly_request else True,
    )

    print(json.dumps({"results": results, "recommended_packs": recommended_packs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
