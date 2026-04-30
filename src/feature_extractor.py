from __future__ import annotations

import numpy as np
import pandas as pd


def _as_float_array(values: list[float] | tuple[float, ...]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def estimate_ocv_from_cycle(cycle: dict, rest_current_threshold: float = 0.02) -> float | None:
    """Estimate OCV from the earliest near-rest point, with a low-current fallback."""
    current = _as_float_array(cycle.get("current_in_A", []))
    voltage = _as_float_array(cycle.get("voltage_in_V", []))
    if len(current) == 0 or len(voltage) == 0 or len(current) != len(voltage):
        return None

    rest_indices = np.where(np.abs(current) <= rest_current_threshold)[0]
    if len(rest_indices) > 0:
        return float(voltage[int(rest_indices[0])])

    min_idx = int(np.argmin(np.abs(current)))
    return float(voltage[min_idx])


def estimate_ir_from_cycle(
    cycle: dict,
    min_current_step: float = 0.5,
    rest_current_threshold: float = 0.1,
    window_size: int = 3,
) -> float | None:
    """
    Estimate DC internal resistance from a current transition near a rest point.

    The HNEI cycles are not all aligned the same way, so we search for either
    rest-to-load or load-to-rest transitions and smooth with local medians.
    """
    current = _as_float_array(cycle.get("current_in_A", []))
    voltage = _as_float_array(cycle.get("voltage_in_V", []))
    if len(current) < (window_size * 2 + 2) or len(current) != len(voltage):
        return None

    candidates: list[tuple[float, int]] = []
    for idx in range(1, len(current)):
        delta_i = current[idx] - current[idx - 1]
        if abs(delta_i) < min_current_step:
            continue

        near_rest = abs(current[idx - 1]) <= rest_current_threshold or abs(current[idx]) <= rest_current_threshold
        score = abs(delta_i) + (1.0 if near_rest else 0.0)
        candidates.append((score, idx))

    if not candidates:
        return None

    for _, idx in sorted(candidates, reverse=True):
        left_start = max(0, idx - window_size)
        left_end = idx
        right_start = idx
        right_end = min(len(current), idx + window_size)
        if left_end <= left_start or right_end <= right_start:
            continue

        left_current = float(np.median(current[left_start:left_end]))
        right_current = float(np.median(current[right_start:right_end]))
        left_voltage = float(np.median(voltage[left_start:left_end]))
        right_voltage = float(np.median(voltage[right_start:right_end]))

        delta_i = right_current - left_current
        if abs(delta_i) < min_current_step:
            continue

        ir_ohm = abs((right_voltage - left_voltage) / delta_i)
        if 0 < ir_ohm < 1:
            return float(ir_ohm)

    return None


def extract_features_per_cell(cell: dict) -> pd.DataFrame:
    nominal_capacity = float(cell["nominal_capacity_in_Ah"])
    rows: list[dict] = []

    for cycle in cell.get("cycle_data", []):
        discharge_capacity = cycle.get("discharge_capacity_in_Ah", [])
        voltage = cycle.get("voltage_in_V", [])

        if not (hasattr(discharge_capacity, "__len__") and len(discharge_capacity) > 0):
            continue
        if not (hasattr(voltage, "__len__") and len(voltage) > 0):
            continue

        capacity_ah = max(discharge_capacity)
        if capacity_ah < 0.1:
            continue

        ir_ohm = estimate_ir_from_cycle(cycle)
        ocv_v = estimate_ocv_from_cycle(cycle)
        soh = float(capacity_ah / nominal_capacity)

        rows.append(
            {
                "cell_id": cell["cell_id"],
                "cycle": int(cycle["cycle_number"]),
                "capacity_Ah": float(capacity_ah),
                "SoH": soh,
                "OCV_V": ocv_v,
                "IR_ohm": ir_ohm,
                "chemistry": cell.get("cathode_material"),
            }
        )

    return pd.DataFrame(rows)


def extract_snapshot_features(cell: dict, at_cycle: int | None = None) -> dict:
    """Take a single snapshot per cell for training or serving analysis."""
    df = extract_features_per_cell(cell)
    if df.empty:
        return {}

    if at_cycle is None:
        row = df.iloc[-1]
    else:
        eligible = df[df["cycle"] <= at_cycle]
        row = eligible.iloc[-1] if not eligible.empty else df.iloc[0]

    return row.to_dict()

