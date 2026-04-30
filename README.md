# ReVoltz Model 2

Cell matcher MVP for reused EV battery cells.

This project trains a model from the HNEI dataset, exposes prediction through FastAPI, and returns:
- outlier flagging for degraded or suspicious cells
- compatibility grouping for repack candidates
- automatic pack assembly recommendations from the compatible cell pool

## Project Layout

```text
.
├── api/                         # FastAPI app
├── artifacts/                   # training outputs
├── data/hnei/                   # HNEI PKL files
├── examples/                    # sample request payloads
├── models/                      # trained model artifact
├── notebooks/                   # notes about optional notebook workflow
├── scripts/                     # local simulation helpers
├── src/                         # training and inference code
├── train.py
├── requirements.txt
└── Dockerfile
```

## Quick Start

Use Python 3.12 if possible.

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv312
./.venv312/bin/pip install -r requirements.txt
./.venv312/bin/python train.py
./.venv312/bin/uvicorn api.main:app --reload --port 8000
```

After the API starts:
- Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Model metadata: `http://localhost:8000/model-info`

## Core AI Flow

The core output is exactly this flow:

1. Workshop sends a list of cells with `cell_id`, `IR_ohm`, and `OCV_V`
2. Model checks whether each cell is outside the training distribution
3. Outliers are marked as `discard`
4. Non-outliers are clustered
5. Cells in the same cluster are filtered again by compatibility thresholds
6. API returns compatible peers for each candidate cell
7. If `assembly_request.target_pack_size` is provided, the system recommends full or partial repack groups

The main endpoint is:

```http
POST /predict-cells
```

Example payload is in [examples/predict_cells_request.json](/Users/rafiffarras/iyref-hackathon/model-2/examples/predict_cells_request.json).

## Simulate User Input

There are two easy ways to simulate the model.

### 1. Through the API

```bash
curl -X POST http://localhost:8000/predict-cells \
  -H "Content-Type: application/json" \
  -d @examples/predict_cells_request.json
```

### 2. Through the local CLI simulator

```bash
./.venv312/bin/python scripts/simulate_prediction.py
./.venv312/bin/python scripts/simulate_prediction.py --input examples/predict_cells_request.json
```

The simulator prints the exact model output without needing the web server.

## React Integration Example

```ts
type CellInput = {
  cell_id: string;
  IR_ohm: number;
  OCV_V: number;
  chemistry?: string | null;
  source_pack_id?: string | null;
};

type PredictRequest = {
  cells: CellInput[];
  assembly_request?: {
    target_pack_size: number;
    max_packs?: number;
    allow_partial_packs?: boolean;
  };
};

export async function predictCells(payload: PredictRequest) {
  const response = await fetch("http://localhost:8000/predict-cells", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Prediction failed: ${response.status}`);
  }

  return response.json();
}
```

Suggested frontend flow:
- form/table input for measured cells
- submit batch to `/predict-cells`
- show `discard` cells in red
- group `candidate` cells by `cluster`
- inside each cluster, highlight `compatible_with`
- render `recommended_packs` as the main repack recommendation cards when pack assembly is requested

## FastAPI Response Shape

Each cell result contains:
- `is_outlier`: whether the cell should be excluded
- `cluster`: cluster ID for non-outliers
- `compatible_with`: list of peer `cell_id`s that passed matching thresholds
- `recommended_pack_id`: repack group assignment when pack assembly is requested
- `status`: `discard`, `candidate`, or `recommended`
- `notes`: human-readable explanation for UI or logs

If `assembly_request` is included, the response also contains:
- `recommended_packs`: full or partial pack suggestions
- each suggestion includes `cell_ids`, `missing_cells`, and `homogeneity_score`

## Training Notes

Training is done in [train.py](/Users/rafiffarras/iyref-hackathon/model-2/train.py), not in a notebook, on purpose:
- easier to reproduce
- easier to automate in CI or deployment
- easier to integrate with backend work

Notebooks are still useful for EDA and experiments, but they are optional for the MVP.

## Current Artifacts

After training, you should have:
- [models/cell_matcher_v1.pkl](/Users/rafiffarras/iyref-hackathon/model-2/models/cell_matcher_v1.pkl)
- [artifacts/training_metadata.json](/Users/rafiffarras/iyref-hackathon/model-2/artifacts/training_metadata.json)
- [artifacts/snapshot_features.csv](/Users/rafiffarras/iyref-hackathon/model-2/artifacts/snapshot_features.csv)
- [artifacts/compatibility_pairs.csv](/Users/rafiffarras/iyref-hackathon/model-2/artifacts/compatibility_pairs.csv)
