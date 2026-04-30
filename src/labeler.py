from __future__ import annotations

from itertools import combinations

import pandas as pd

from src.config import COMPATIBILITY_IR_THRESHOLD, COMPATIBILITY_OCV_THRESHOLD


def generate_compatibility_pairs(
    df_snapshots: pd.DataFrame,
    ir_threshold: float = COMPATIBILITY_IR_THRESHOLD,
    ocv_threshold: float = COMPATIBILITY_OCV_THRESHOLD,
) -> pd.DataFrame:
    """
    Create pairwise compatibility labels from snapshot features.

    Chemistry matching is enforced only when both values are present.
    """
    pairs: list[dict] = []
    cell_ids = df_snapshots["cell_id"].unique()

    for cell_a, cell_b in combinations(cell_ids, 2):
        a = df_snapshots[df_snapshots["cell_id"] == cell_a].iloc[0]
        b = df_snapshots[df_snapshots["cell_id"] == cell_b].iloc[0]

        if pd.isna(a["IR_ohm"]) or pd.isna(b["IR_ohm"]):
            continue
        if pd.isna(a["OCV_V"]) or pd.isna(b["OCV_V"]):
            continue

        delta_ir = abs(float(a["IR_ohm"]) - float(b["IR_ohm"]))
        delta_ocv = abs(float(a["OCV_V"]) - float(b["OCV_V"]))
        same_chemistry = True
        if a.get("chemistry") and b.get("chemistry"):
            same_chemistry = a["chemistry"] == b["chemistry"]

        compatible = int(delta_ir < ir_threshold and delta_ocv < ocv_threshold and same_chemistry)
        pairs.append(
            {
                "cell_a": cell_a,
                "cell_b": cell_b,
                "IR_a": float(a["IR_ohm"]),
                "IR_b": float(b["IR_ohm"]),
                "OCV_a": float(a["OCV_V"]),
                "OCV_b": float(b["OCV_V"]),
                "delta_IR": delta_ir,
                "delta_OCV": delta_ocv,
                "same_chemistry": same_chemistry,
                "compatible": compatible,
            }
        )

    return pd.DataFrame(pairs)

