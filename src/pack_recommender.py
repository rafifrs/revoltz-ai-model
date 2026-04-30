from __future__ import annotations

from collections import defaultdict
from itertools import combinations


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def _pair_metrics(left: dict, right: dict) -> tuple[float, float]:
    return abs(left["IR_ohm"] - right["IR_ohm"]), abs(left["OCV_V"] - right["OCV_V"])


def _pair_score(left: dict, right: dict, ir_threshold: float, ocv_threshold: float) -> float:
    delta_ir, delta_ocv = _pair_metrics(left, right)
    ir_component = max(0.0, 1.0 - (delta_ir / ir_threshold))
    ocv_component = max(0.0, 1.0 - (delta_ocv / ocv_threshold))
    return (ir_component + ocv_component) / 2.0


def _group_score(group: list[dict], ir_threshold: float, ocv_threshold: float) -> float:
    if len(group) < 2:
        return 0.0
    scores = [_pair_score(left, right, ir_threshold, ocv_threshold) for left, right in combinations(group, 2)]
    return sum(scores) / len(scores)


def _group_pairwise_averages(group: list[dict]) -> tuple[float, float]:
    if len(group) < 2:
        return 0.0, 0.0
    delta_irs: list[float] = []
    delta_ocvs: list[float] = []
    for left, right in combinations(group, 2):
        delta_ir, delta_ocv = _pair_metrics(left, right)
        delta_irs.append(delta_ir)
        delta_ocvs.append(delta_ocv)
    return sum(delta_irs) / len(delta_irs), sum(delta_ocvs) / len(delta_ocvs)


def recommend_packs(
    results: list[dict],
    target_pack_size: int,
    ir_threshold: float,
    ocv_threshold: float,
    max_packs: int | None = None,
    allow_partial_packs: bool = True,
) -> list[dict]:
    """
    Build pack recommendations from compatible cells.

    The strategy is greedy and clique-based:
    - only non-outlier cells are eligible
    - a recommended pack only contains cells that are mutually compatible
    - we fill packs starting from the highest-degree nodes first
    """
    eligible_cells = [cell for cell in results if not cell["is_outlier"]]
    cells_by_id = {cell["cell_id"]: cell for cell in eligible_cells}
    adjacency: dict[str, set[str]] = {
        cell["cell_id"]: {peer for peer in cell.get("compatible_with", []) if peer in cells_by_id}
        for cell in eligible_cells
    }

    used: set[str] = set()
    recommendations: list[dict] = []
    pack_index = 1

    cluster_to_cells: dict[int | None, list[str]] = defaultdict(list)
    for cell in eligible_cells:
        cluster_to_cells[cell.get("cluster")].append(cell["cell_id"])

    for cluster, cluster_cell_ids in sorted(cluster_to_cells.items(), key=lambda item: (item[0] is None, item[0])):
        candidate_ids = [cell_id for cell_id in cluster_cell_ids if cell_id not in used]
        candidate_ids.sort(key=lambda cell_id: (-len(adjacency[cell_id]), cell_id))

        while candidate_ids:
            seed_id = candidate_ids.pop(0)
            if seed_id in used:
                candidate_ids = [cell_id for cell_id in candidate_ids if cell_id not in used]
                continue

            group_ids = [seed_id]
            available_ids = [cell_id for cell_id in candidate_ids if cell_id not in used]

            while len(group_ids) < target_pack_size:
                compatible_candidates = [
                    cell_id
                    for cell_id in available_ids
                    if all(
                        cell_id in adjacency[member_id] and member_id in adjacency[cell_id]
                        for member_id in group_ids
                    )
                ]
                if not compatible_candidates:
                    break

                compatible_candidates.sort(
                    key=lambda cell_id: (
                        -sum(
                            _pair_score(cells_by_id[cell_id], cells_by_id[member_id], ir_threshold, ocv_threshold)
                            for member_id in group_ids
                        ),
                        cell_id,
                    )
                )
                chosen_id = compatible_candidates[0]
                group_ids.append(chosen_id)
                available_ids.remove(chosen_id)

            if len(group_ids) == target_pack_size or (allow_partial_packs and len(group_ids) > 1):
                group = [cells_by_id[cell_id] for cell_id in group_ids]
                avg_ir = sum(cell["IR_ohm"] for cell in group) / len(group)
                avg_ocv = sum(cell["OCV_V"] for cell in group) / len(group)
                avg_delta_ir, avg_delta_ocv = _group_pairwise_averages(group)
                homogeneity_score = _group_score(group, ir_threshold, ocv_threshold)
                missing_cells = max(0, target_pack_size - len(group_ids))
                status = "full" if missing_cells == 0 else "partial"

                notes = []
                if status == "full":
                    notes.append("All member cells are mutually compatible and meet the requested pack size.")
                else:
                    notes.append("This is the strongest mutually compatible partial group found in the current cell pool.")

                recommendations.append(
                    {
                        "pack_id": f"pack_{pack_index}",
                        "status": status,
                        "cluster": cluster,
                        "cell_ids": group_ids,
                        "source_pack_ids": sorted({cell["source_pack_id"] for cell in group if cell.get("source_pack_id")}),
                        "target_pack_size": target_pack_size,
                        "cell_count": len(group_ids),
                        "missing_cells": missing_cells,
                        "average_ir_ohm": avg_ir,
                        "average_ocv_v": avg_ocv,
                        "average_pairwise_delta_ir": avg_delta_ir,
                        "average_pairwise_delta_ocv": avg_delta_ocv,
                        "homogeneity_score": homogeneity_score,
                        "notes": notes,
                    }
                )
                pack_index += 1
                used.update(group_ids)

            candidate_ids = [cell_id for cell_id in candidate_ids if cell_id not in used]
            if max_packs is not None and len(recommendations) >= max_packs:
                break

        if max_packs is not None and len(recommendations) >= max_packs:
            break

    recommendation_by_cell = {}
    for recommendation in recommendations:
        for cell_id in recommendation["cell_ids"]:
            recommendation_by_cell[cell_id] = recommendation["pack_id"]

    for cell in results:
        cell["recommended_pack_id"] = recommendation_by_cell.get(cell["cell_id"])
        if cell["recommended_pack_id"]:
            if cell["status"] == "candidate":
                cell["status"] = "recommended"
            cell["notes"].append(f"Assigned to {cell['recommended_pack_id']} for repack recommendation.")

    return recommendations

