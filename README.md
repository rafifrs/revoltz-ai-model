# ReVoltz AI Model

Battery intelligence engine for ReVoltz — an EV battery second-life platform. This repository contains two production models exposed through a unified FastAPI service:

- **Model 1 — SoH Predictor:** Predicts State-of-Health for a full battery pack from measured electrical and usage parameters, then recommends whether to recondition, repack, or recycle it.
- **Model 2 — Cell Matcher:** Screens individual harvested cells, flags degraded outliers, clusters compatible cells together, and assembles optimal repack groups.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Layout](#project-layout)
3. [Quick Start](#quick-start)
4. [Running with Docker](#running-with-docker)
5. [API Reference](#api-reference)
6. [Request & Response Shapes](#request--response-shapes)
7. [Simulate Without the Server](#simulate-without-the-server)
8. [Frontend Integration](#frontend-integration)
9. [Training](#training)
10. [Model Internals](#model-internals)
11. [Configuration](#configuration)
12. [Current Artifacts](#current-artifacts)

---

## Architecture Overview

```
Workshop Input
      │
      ├── Full Pack Measurements ──► POST /predict-pack ──► SoH score + action recommendation
      │                                  (Model 1: XGBoost / LightGBM regression)
      │
      └── Individual Cell Measurements ► POST /predict-cells ──► Per-cell status + pack groups
                                             (Model 2: IQR outlier detection + K-Means clustering)
```

A workshop operator inputs measured values directly from a multimeter and IR meter. The API returns actionable decisions with no post-processing required on the client side.

---

## Project Layout

```
.
├── api/
│   ├── main.py                 # FastAPI app, endpoints, lifespan
│   └── schema.py               # Pydantic request/response models
├── artifacts/
│   ├── compatibility_pairs.csv     # generated during training
│   ├── snapshot_features.csv       # cell snapshot features (training set)
│   ├── snapshot_features_valid.csv # cell snapshot features (validation set)
│   └── training_metadata.json      # model parameters and evaluation scores
├── examples/
│   └── predict_cells_request.json  # sample payload for /predict-cells
├── models/
│   ├── cell_matcher_v1.pkl         # trained Model 2 artifact
│   └── soh_predictor_v1.pkl        # trained Model 1 artifact
├── notebooks/
│   ├── cell_matcher.ipynb          # EDA and Model 2 experiments
│   └── soh_predictor.ipynb         # EDA and Model 1 experiments
├── scripts/
│   ├── train.py                    # full training pipeline (both models)
│   └── simulate_prediction.py      # CLI simulator for /predict-cells
├── src/
│   ├── config.py                   # paths, thresholds, constants
│   ├── data_loader.py              # HNEI .pkl parser
│   ├── feature_extractor.py        # DCIR extraction from voltage curves
│   ├── labeler.py                  # compatibility pair generation
│   ├── model1_training.py          # SoH predictor training logic
│   ├── pack_recommender.py         # pack assembly algorithm
│   ├── predictor.py                # inference entry points for both models
│   └── training.py                 # cell matcher training logic
├── requirements.txt
└── Dockerfile
```

---

## Quick Start

Python 3.10 or 3.12 is required.

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train both models (requires data — see Training section)
python scripts/train.py

# Start the API server
uvicorn api.main:app --reload --port 8000
```

Once running:

| Endpoint | URL |
|---|---|
| Swagger UI | `http://localhost:8000/docs` |
| Health check | `http://localhost:8000/health` |
| Model 2 metadata | `http://localhost:8000/model-info` |
| Model 1 metadata | `http://localhost:8000/model1-info` |

---

## Running with Docker

```bash
docker build -t revoltz-ai-model .
docker run -p 8000:8000 revoltz-ai-model
```

To configure allowed CORS origins at runtime:

```bash
docker run -p 8000:8000 -e ALLOWED_ORIGINS="https://app.revoltz.id,https://staging.revoltz.id" revoltz-ai-model
```

---

## API Reference

### `POST /predict-cells`

Screen a batch of individual cells from a disassembled pack. Optionally request automatic repack group assembly.

### `POST /predict-pack`

Predict State-of-Health for a full battery pack and get a recommended action.

### `GET /model-info`

Returns training parameters and evaluation scores for the Cell Matcher (Model 2).

### `GET /model1-info`

Returns training parameters and evaluation scores for the SoH Predictor (Model 1).

### `GET /health`

Returns `{"status": "healthy"}`. Use for container health checks.

---

## Request & Response Shapes

### `POST /predict-cells`

**Request:**

```json
{
  "cells": [
    {
      "cell_id": "workshop_cell_A",
      "IR_ohm": 0.075,
      "OCV_V": 3.0,
      "source_pack_id": "pack_alpha",
      "chemistry": "NMC",
      "source_battery_id": "battery_001",
      "position_in_pack": "slot_3"
    }
  ],
  "assembly_request": {
    "target_pack_size": 3,
    "max_packs": 10,
    "allow_partial_packs": true
  }
}
```

- `cells`: 1–500 cells per request
- `assembly_request`: optional — include to get `recommended_packs` in the response
- `chemistry`, `source_pack_id`, `source_battery_id`, `position_in_pack`: all optional metadata

**Response:**

```json
{
  "results": [
    {
      "cell_id": "workshop_cell_A",
      "IR_ohm": 0.075,
      "OCV_V": 3.0,
      "is_outlier": false,
      "cluster": 1,
      "compatible_with": ["workshop_cell_B", "workshop_cell_C"],
      "recommended_pack_id": "pack_rec_0",
      "status": "recommended",
      "notes": ["Cell passed outlier check", "Assigned to repack group pack_rec_0"]
    }
  ],
  "summary": {
    "total_cells": 4,
    "outlier_count": 1,
    "valid_for_repack": 3,
    "clusters_found": 2,
    "compatible_pairs": 3,
    "recommended_pack_count": 1,
    "repack_rate": 0.75,
    "status_breakdown": {
      "recommended": 3,
      "discard": 1
    }
  },
  "recommended_packs": [
    {
      "pack_id": "pack_rec_0",
      "status": "full",
      "cluster": 1,
      "cell_ids": ["workshop_cell_A", "workshop_cell_B", "workshop_cell_C"],
      "target_pack_size": 3,
      "cell_count": 3,
      "missing_cells": 0,
      "average_ir_ohm": 0.074,
      "average_ocv_v": 3.01,
      "average_pairwise_delta_ir": 0.003,
      "average_pairwise_delta_ocv": 0.01,
      "homogeneity_score": 0.94,
      "notes": []
    }
  ]
}
```

Cell `status` values:

| Status | Meaning |
|---|---|
| `discard` | Outlier — IR or OCV outside training distribution. Do not repack. |
| `candidate` | Passed outlier check and clustered, but not assigned to a pack group. |
| `recommended` | Assigned to a full or partial repack group. |

---

### `POST /predict-pack`

**Request:**

```json
{
  "packs": [
    {
      "pack_id": "pack_001",
      "ocv_v": 48.5,
      "capacity_ah": 45.2,
      "cycle_count": 620,
      "temperature_c": 25.0,
      "age_days": 730,
      "chemistry": "NMC"
    }
  ]
}
```

**Response:**

```json
{
  "results": [
    {
      "pack_id": "pack_001",
      "predicted_soh": 0.83,
      "confidence_score": 0.91,
      "recommended_action": "recondition",
      "notes": ["SoH above recondition threshold (0.80)"]
    }
  ],
  "summary": {
    "total_packs": 1,
    "average_predicted_soh": 0.83,
    "average_confidence_score": 0.91,
    "recondition_count": 1,
    "repack_count": 0,
    "recycle_count": 0
  }
}
```

`recommended_action` values: `recondition` (SoH ≥ 0.80), `repack` (SoH ≥ 0.60), `recycle` (SoH < 0.60).

---

## Simulate Without the Server

Test the cell matcher inference pipeline locally without starting the API:

```bash
# Use built-in sample data
python scripts/simulate_prediction.py

# Use a custom payload
python scripts/simulate_prediction.py --input examples/predict_cells_request.json
```

The simulator prints the full model output to stdout.

---

## Frontend Integration

### TypeScript client for `/predict-cells`

```typescript
type CellInput = {
  cell_id: string;
  IR_ohm: number;
  OCV_V: number;
  chemistry?: string | null;
  source_pack_id?: string | null;
  source_battery_id?: string | null;
  position_in_pack?: string | null;
};

type AssemblyRequest = {
  target_pack_size: number;
  max_packs?: number;
  allow_partial_packs?: boolean;
};

type PredictCellsRequest = {
  cells: CellInput[];
  assembly_request?: AssemblyRequest;
};

export async function predictCells(payload: PredictCellsRequest) {
  const response = await fetch("/predict-cells", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Prediction failed: ${response.status}`);
  }

  return response.json();
}
```

Suggested UI rendering logic:

- Show cells where `status === "discard"` in red with the `notes` field as tooltip
- Group `candidate` and `recommended` cells by `cluster`
- Highlight `compatible_with` peers on hover
- Render each entry in `recommended_packs` as a repack group card, showing `homogeneity_score` and `missing_cells`

---

## Training

### Dataset

Both models are trained on publicly available battery datasets:

| Model | Dataset | Source |
|---|---|---|
| Cell Matcher (Model 2) | HNEI 18650 NMC-LCO | [HuggingFace — BatteryLife](https://huggingface.co/datasets/Ruifeng-Tan/BatteryLife) |
| SoH Predictor (Model 1) | NASA Battery Dataset | NASA Prognostics Data Repository |

The HNEI dataset contains 14 cells with 1,000+ charge/discharge cycles each. Internal resistance is not stored directly — it is extracted from the voltage/current time-series using the DCIR method (see [Model Internals](#model-internals)).

### Running Training

Place HNEI `.pkl` files in `data/hnei/` and NASA data in `data/nasa_soh_dataset/`, then:

```bash
python scripts/train.py
```

This produces:
- `models/cell_matcher_v1.pkl`
- `models/soh_predictor_v1.pkl`
- `artifacts/training_metadata.json`
- `artifacts/snapshot_features.csv`
- `artifacts/compatibility_pairs.csv`

Training is intentionally script-based rather than notebook-based for reproducibility and CI compatibility. Notebooks in `notebooks/` are for EDA and offline experimentation only.

---

## Model Internals

### Model 2 — Cell Matcher

**Step 1: Feature Extraction**

The HNEI dataset stores `internal_resistance_in_ohm` as `None` for all cycles. IR is computed using **DC Internal Resistance (DCIR)**:

```
IR = ΔV / ΔI
```

At each cycle's discharge onset — the moment current transitions from rest to discharge — the voltage drop divided by the current step gives the IR estimate. Open Circuit Voltage (OCV) is taken from the first sample where current is approximately zero.

A snapshot at cycle 100 is taken per cell as the representative training feature.

**Step 2: Outlier Detection**

IQR-based bounds are computed from the training distribution:

```
lower = Q1 − 1.5 × IQR
upper = Q3 + 1.5 × IQR
```

Any cell submitted at inference whose IR or OCV falls outside these bounds is immediately flagged `discard`. Current trained bounds:

| Feature | Lower | Upper |
|---|---|---|
| IR (Ohm) | 0.0688 | 0.0855 |
| OCV (V) | 2.294 | 4.175 |

**Step 3: K-Means Clustering**

Non-outlier cells are scaled with `StandardScaler` and clustered using K-Means. The number of clusters is selected by maximising silhouette score over k = 2..6. The current model uses **k = 2**, achieving a silhouette score of **0.477**.

**Step 4: Compatibility Filtering**

Within each cluster, cells are additionally filtered by pairwise electrochemical thresholds:

| Metric | Threshold |
|---|---|
| `|IR_A − IR_B|` | < 5 mΩ (0.005 Ohm) |
| `|OCV_A − OCV_B|` | < 50 mV (0.05 V) |

These thresholds follow established BMS literature for series cell matching.

**Step 5: Pack Assembly**

The `PackRecommender` greedily assembles cells from the same cluster into groups of `target_pack_size`, ranked by internal homogeneity score. Partial groups are included when `allow_partial_packs` is true.

### Model 1 — SoH Predictor

Trained on pack-level features (OCV, capacity, cycle count, temperature, age, chemistry) using gradient-boosted regression. The model outputs a continuous SoH score in [0, 1]. Action thresholds:

| Predicted SoH | Recommended Action |
|---|---|
| ≥ 0.80 | `recondition` |
| 0.60 – 0.79 | `repack` |
| < 0.60 | `recycle` |

---

## Configuration

All thresholds and paths are centralised in [src/config.py](src/config.py):

| Constant | Default | Description |
|---|---|---|
| `COMPATIBILITY_IR_THRESHOLD` | `0.005` | Max allowed IR delta between compatible cells (Ohm) |
| `COMPATIBILITY_OCV_THRESHOLD` | `0.05` | Max allowed OCV delta between compatible cells (V) |
| `IQR_MULTIPLIER` | `1.5` | Multiplier for IQR outlier bounds |
| `DEFAULT_SNAPSHOT_CYCLE` | `100` | Training snapshot cycle index |
| `MODEL1_RECONDITION_MIN_SOH` | `0.80` | SoH threshold for recondition recommendation |
| `MODEL1_REPACK_MIN_SOH` | `0.60` | SoH threshold for repack recommendation |

CORS origins are controlled at runtime via the `ALLOWED_ORIGINS` environment variable (comma-separated). Defaults to `*` when unset.

---

## Current Artifacts

```
artifacts/
├── training_metadata.json          model version, cluster count, silhouette score, IQR bounds
├── snapshot_features.csv           per-cell IR/OCV/SoH snapshot (training set, 14 cells)
├── snapshot_features_valid.csv     per-cell IR/OCV/SoH snapshot (validation set)
└── compatibility_pairs.csv         all pairwise compatibility labels generated during training
```

Current `cell_matcher_v1` training summary:

| Parameter | Value |
|---|---|
| Training cells | 14 |
| Non-outlier cells | 9 |
| Clusters (k) | 2 |
| Silhouette score | 0.477 |
| IR compatibility threshold | 5 mΩ |
| OCV compatibility threshold | 50 mV |
| Snapshot cycle | 100 |

---

*ReVoltz — AI-driven EV Battery Decision System*
