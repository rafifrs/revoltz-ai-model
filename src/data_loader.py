from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_cell(filepath: str | Path) -> dict:
    """Load a single HNEI PKL file."""
    path = Path(filepath)
    with path.open("rb") as handle:
        return pickle.load(handle)


def iter_cell_files(data_dir: str | Path) -> Iterable[Path]:
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".pkl"):
            yield path / fname


def load_all_cells(data_dir: str | Path) -> list[dict]:
    cells: list[dict] = []
    for filepath in iter_cell_files(data_dir):
        cells.append(load_cell(filepath))
    return cells


def extract_capacity_fade(cell: dict) -> pd.DataFrame:
    """Extract cycle-level discharge capacity and SoH trajectory."""
    nominal = float(cell["nominal_capacity_in_Ah"])
    rows: list[dict] = []

    for cycle in cell.get("cycle_data", []):
        discharge_capacity = cycle.get("discharge_capacity_in_Ah", [])
        if hasattr(discharge_capacity, "__len__") and len(discharge_capacity) > 0:
            capacity = max(discharge_capacity)
            if capacity > 0.1:
                rows.append(
                    {
                        "cell_id": cell["cell_id"],
                        "cycle": int(cycle["cycle_number"]),
                        "capacity_Ah": float(capacity),
                        "SoH": float(capacity / nominal),
                    }
                )

    return pd.DataFrame(rows)

