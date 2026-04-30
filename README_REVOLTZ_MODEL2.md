# ReVoltz — Model 2: Cell Matcher
### End-to-End Development Guide (VSCode)

> **Tujuan Model 2:** Menerima input voltage dan internal resistance per sel individual dari bengkel, lalu menghasilkan keputusan: sel mana yang kompatibel untuk direpack bersama, dan sel mana yang harus dibuang (outlier/degraded).

---

## Daftar Isi

1. [Struktur Folder](#1-struktur-folder)
2. [Setup Environment](#2-setup-environment)
3. [Dataset & Sumber Data](#3-dataset--sumber-data)
4. [Eksplorasi Data (EDA)](#4-eksplorasi-data-eda)
5. [Feature Engineering — Ekstraksi IR dari Voltage Curve](#5-feature-engineering--ekstraksi-ir-dari-voltage-curve)
6. [Pembuatan Label Kompatibilitas](#6-pembuatan-label-kompatibilitas)
7. [Training Model](#7-training-model)
8. [Evaluasi Model](#8-evaluasi-model)
9. [Export & Serving Model](#9-export--serving-model)
10. [API Layer dengan FastAPI](#10-api-layer-dengan-fastapi)
11. [Deploy ke Railway / Render](#11-deploy-ke-railway--render)
12. [Catatan Teknis Penting](#12-catatan-teknis-penting)

---

## 1. Struktur Folder

```
revoltz-model2/
├── data/
│   ├── hnei/                          # 14 file .pkl dari HuggingFace BatteryLife
│   │   ├── HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_a.pkl
│   │   ├── HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_b.pkl
│   │   └── ... (total 14 file)
│   └── stanford_parallel/             # dari Mendeley Data (opsional, tambahan)
├── notebooks/
│   ├── 01_eda.ipynb                   # eksplorasi data
│   ├── 02_feature_engineering.ipynb   # ekstraksi fitur IR dari voltage
│   └── 03_modeling.ipynb              # training & evaluasi
├── src/
│   ├── data_loader.py                 # load & parse semua .pkl
│   ├── feature_extractor.py           # ekstraksi IR dari voltage/current
│   ├── labeler.py                     # generate label kompatibilitas
│   ├── model.py                       # outlier flagging + k-means
│   └── predictor.py                   # fungsi predict untuk API
├── api/
│   ├── main.py                        # FastAPI app
│   └── schema.py                      # Pydantic input/output schema
├── models/
│   └── cell_matcher_v1.pkl            # model tersimpan setelah training
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 2. Setup Environment

### Prasyarat
- Python 3.10+
- VSCode dengan extension: Python, Jupyter, Pylance

### Buat virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### Install dependencies

Buat file `requirements.txt`:

```
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
matplotlib==3.8.4
seaborn==0.13.2
jupyter==1.0.0
fastapi==0.111.0
uvicorn==0.29.0
pydantic==2.7.1
pickle5==0.0.12
```

Lalu install:

```bash
pip install -r requirements.txt
```

---

## 3. Dataset & Sumber Data

### HNEI Dataset (Primer)
- **Sumber:** HuggingFace — BatteryLife Processed / HNEI
- **Download:** https://huggingface.co/datasets/Ruifeng-Tan/BatteryLife
- **Format:** `.pkl` per sel (14 file, masing-masing ~27–31 MB)
- **Chemistry:** NMC-LCO, 18650 cylindrical
- **Isi per file:** 1 sel = ~1000+ cycles, berisi voltage, current, capacity per cycle

### Struktur Data PKL (hasil eksplorasi aktual)

Setiap file `.pkl` adalah dictionary dengan struktur:

```python
{
  'cell_id': str,                        # ID unik sel, contoh: "HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_a"
  'nominal_capacity_in_Ah': float,       # kapasitas nominal, contoh: 2.8 Ah
  'cathode_material': str,               # contoh: "LiCoO2+LiNi0.4Co0.4Mn0.2O2"
  'cycle_data': list[dict],              # list 1000+ cycle, inti data
  ...
}
```

Setiap elemen `cycle_data` berisi:

```python
{
  'cycle_number': int,
  'current_in_A': list[float],           # time-series arus per cycle
  'voltage_in_V': list[float],           # time-series voltage per cycle
  'charge_capacity_in_Ah': list[float],
  'discharge_capacity_in_Ah': list[float],
  'time_in_s': list[float],
  'temperature_in_C': list[float],
  'internal_resistance_in_ohm': None     # ⚠️ KOSONG — harus diekstrak dari voltage/current
}
```

> **Catatan penting:** Field `internal_resistance_in_ohm` adalah None di semua cycle untuk dataset HNEI di BatteryLife. IR harus dihitung manual dari voltage drop saat pulse current — lihat Section 5.

---

## 4. Eksplorasi Data (EDA)

Buat file `src/data_loader.py`:

```python
import pickle
import pandas as pd
import os

def load_cell(filepath: str) -> dict:
    """Load satu file .pkl sel HNEI"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_all_cells(data_dir: str) -> list[dict]:
    """Load semua file .pkl dari folder"""
    cells = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith('.pkl'):
            cell = load_cell(os.path.join(data_dir, fname))
            cells.append(cell)
            print(f"Loaded: {cell['cell_id']} — {len(cell['cycle_data'])} cycles")
    return cells

def extract_capacity_fade(cell: dict) -> pd.DataFrame:
    """Ekstrak kapasitas per cycle sebagai SoH proxy"""
    nominal = cell['nominal_capacity_in_Ah']
    rows = []
    for c in cell['cycle_data']:
        dc = c['discharge_capacity_in_Ah']
        if hasattr(dc, '__len__') and len(dc) > 0:
            max_cap = max(dc)
            if max_cap > 0.1:  # filter noise
                rows.append({
                    'cell_id': cell['cell_id'],
                    'cycle': c['cycle_number'],
                    'capacity_Ah': max_cap,
                    'SoH': max_cap / nominal
                })
    return pd.DataFrame(rows)
```

Gunakan di notebook `01_eda.ipynb`:

```python
from src.data_loader import load_all_cells, extract_capacity_fade
import matplotlib.pyplot as plt

cells = load_all_cells('data/hnei/')

# Plot capacity fade semua sel
fig, ax = plt.subplots(figsize=(12, 6))
for cell in cells:
    df = extract_capacity_fade(cell)
    ax.plot(df['cycle'], df['SoH'], label=cell['cell_id'][-1], alpha=0.7)

ax.set_xlabel('Cycle')
ax.set_ylabel('SoH')
ax.set_title('Cell-to-Cell SoH Variation — HNEI Dataset')
ax.legend()
plt.tight_layout()
plt.show()
```

---

## 5. Feature Engineering — Ekstraksi IR dari Voltage Curve

Karena IR tidak tersedia langsung, gunakan metode **DC Internal Resistance (DCIR)** dari voltage pulse:

```
IR = ΔV / ΔI
```

Ambil momen awal discharge di setiap cycle: saat current berubah dari 0 ke nilai discharge, voltage akan drop. Rasio voltage drop terhadap current step = IR estimasi.

Buat file `src/feature_extractor.py`:

```python
import numpy as np
import pandas as pd

def estimate_ir_from_cycle(cycle: dict) -> float | None:
    """
    Estimasi internal resistance dari voltage drop saat awal discharge.
    Metode: DCIR = delta_V / delta_I pada transisi current step.
    Return: IR dalam Ohm, atau None jika tidak bisa dihitung.
    """
    current = np.array(cycle['current_in_A'])
    voltage = np.array(cycle['voltage_in_V'])

    if len(current) < 5:
        return None

    # Cari indeks transisi: dari ~0A ke discharge current
    for i in range(1, len(current) - 1):
        delta_I = abs(current[i] - current[i - 1])
        if delta_I > 0.5:  # threshold: ada perubahan arus signifikan
            delta_V = abs(voltage[i] - voltage[i - 1])
            if delta_I > 0:
                return delta_V / delta_I
    return None

def extract_features_per_cell(cell: dict) -> pd.DataFrame:
    """
    Ekstrak fitur per cycle dari satu sel.
    Output: DataFrame dengan kolom cell_id, cycle, SoH, IR_ohm, OCV_V, capacity_Ah
    """
    nominal = cell['nominal_capacity_in_Ah']
    rows = []

    for c in cell['cycle_data']:
        dc = c['discharge_capacity_in_Ah']
        v = c['voltage_in_V']

        if not (hasattr(dc, '__len__') and len(dc) > 0):
            continue

        max_cap = max(dc)
        if max_cap < 0.1:
            continue

        # OCV: ambil voltage di awal cycle sebelum current mengalir
        current = c['current_in_A']
        ocv = None
        for i, curr in enumerate(current):
            if abs(curr) < 0.01 and i < 20:  # arus hampir nol = kondisi istirahat
                ocv = v[i]
                break

        ir = estimate_ir_from_cycle(c)
        soh = max_cap / nominal

        rows.append({
            'cell_id': cell['cell_id'],
            'cycle': c['cycle_number'],
            'capacity_Ah': max_cap,
            'SoH': soh,
            'OCV_V': ocv,
            'IR_ohm': ir,
        })

    return pd.DataFrame(rows)

def extract_snapshot_features(cell: dict, at_cycle: int = None) -> dict:
    """
    Ambil snapshot fitur sel di satu titik cycle tertentu (atau cycle terakhir).
    Ini yang dipakai saat bengkel input data satu sel.
    """
    df = extract_features_per_cell(cell)
    if df.empty:
        return {}

    if at_cycle:
        row = df[df['cycle'] <= at_cycle].iloc[-1]
    else:
        row = df.iloc[-1]  # cycle terakhir

    return row.to_dict()
```

---

## 6. Pembuatan Label Kompatibilitas

Tidak ada dataset publik yang punya label "sel A kompatibel dengan sel B". Label dibuat dari **rule elektrokimia yang well-established**:

> Dua sel dianggap **kompatibel** jika:
> - `|IR_A - IR_B| < 5 mΩ` (0.005 Ohm)
> - `|OCV_A - OCV_B| < 50 mV` (0.05 V)
> - Chemistry sama

Buat file `src/labeler.py`:

```python
import pandas as pd
import numpy as np
from itertools import combinations

def generate_compatibility_pairs(df_snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    Dari DataFrame snapshot semua sel, generate pasangan sel
    dengan label compatible (1) atau not compatible (0).

    Input df_snapshots kolom: cell_id, IR_ohm, OCV_V, SoH
    """
    IR_THRESHOLD = 0.005   # 5 mΩ
    OCV_THRESHOLD = 0.05   # 50 mV

    pairs = []
    cell_ids = df_snapshots['cell_id'].unique()

    for id_a, id_b in combinations(cell_ids, 2):
        a = df_snapshots[df_snapshots['cell_id'] == id_a].iloc[0]
        b = df_snapshots[df_snapshots['cell_id'] == id_b].iloc[0]

        if pd.isna(a['IR_ohm']) or pd.isna(b['IR_ohm']):
            continue
        if pd.isna(a['OCV_V']) or pd.isna(b['OCV_V']):
            continue

        delta_IR = abs(a['IR_ohm'] - b['IR_ohm'])
        delta_OCV = abs(a['OCV_V'] - b['OCV_V'])

        compatible = int(delta_IR < IR_THRESHOLD and delta_OCV < OCV_THRESHOLD)

        pairs.append({
            'cell_a': id_a,
            'cell_b': id_b,
            'IR_a': a['IR_ohm'],
            'IR_b': b['IR_ohm'],
            'OCV_a': a['OCV_V'],
            'OCV_b': b['OCV_V'],
            'delta_IR': delta_IR,
            'delta_OCV': delta_OCV,
            'compatible': compatible
        })

    return pd.DataFrame(pairs)
```

---

## 7. Training Model

Model 2 terdiri dari dua komponen yang berjalan berurutan:

### Step 1 — IQR Outlier Flagging

Identifikasi sel yang IR atau OCV-nya anomali dibanding populasi. Sel outlier langsung diflag sebagai "degraded" dan tidak masuk ke clustering.

### Step 2 — K-Means Clustering

Sel-sel yang lolos outlier check dikelompokkan berdasarkan kemiripan IR dan OCV. Sel dalam cluster yang sama = kandidat kompatibel untuk direpack.

Buat file `src/model.py`:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle

class CellMatcher:
    def __init__(self, n_clusters: int = 3, iqr_multiplier: float = 1.5):
        self.n_clusters = n_clusters
        self.iqr_multiplier = iqr_multiplier
        self.scaler = StandardScaler()
        self.kmeans = None
        self.ir_bounds = None
        self.ocv_bounds = None

    def _compute_iqr_bounds(self, values: np.ndarray):
        """Hitung batas bawah dan atas IQR untuk deteksi outlier"""
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower = Q1 - self.iqr_multiplier * IQR
        upper = Q3 + self.iqr_multiplier * IQR
        return lower, upper

    def fit(self, df: pd.DataFrame):
        """
        Training model dari DataFrame dengan kolom: IR_ohm, OCV_V
        """
        # Hitung bounds IQR dari data training
        self.ir_bounds = self._compute_iqr_bounds(df['IR_ohm'].dropna().values)
        self.ocv_bounds = self._compute_iqr_bounds(df['OCV_V'].dropna().values)

        # Filter: hanya sel non-outlier untuk training K-Means
        mask = (
            df['IR_ohm'].between(*self.ir_bounds) &
            df['OCV_V'].between(*self.ocv_bounds)
        )
        df_clean = df[mask].copy()

        features = df_clean[['IR_ohm', 'OCV_V']].values
        features_scaled = self.scaler.fit_transform(features)

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(features_scaled)

        score = silhouette_score(features_scaled, self.kmeans.labels_)
        print(f"Training selesai. Silhouette score: {score:.4f}")
        return self

    def predict(self, cells: list[dict]) -> list[dict]:
        """
        Input: list of dict, masing-masing {'cell_id': str, 'IR_ohm': float, 'OCV_V': float}
        Output: list of dict dengan tambahan 'is_outlier', 'cluster', 'compatible_with'
        """
        results = []
        valid_cells = []

        for cell in cells:
            ir = cell.get('IR_ohm')
            ocv = cell.get('OCV_V')

            # Outlier check
            is_outlier = (
                ir is None or ocv is None or
                not (self.ir_bounds[0] <= ir <= self.ir_bounds[1]) or
                not (self.ocv_bounds[0] <= ocv <= self.ocv_bounds[1])
            )

            result = {**cell, 'is_outlier': is_outlier, 'cluster': None}
            results.append(result)

            if not is_outlier:
                valid_cells.append((len(results) - 1, ir, ocv))

        # K-Means clustering untuk sel yang lolos
        if valid_cells and self.kmeans:
            indices, ir_vals, ocv_vals = zip(*valid_cells)
            features = np.array([[ir, ocv] for ir, ocv in zip(ir_vals, ocv_vals)])
            features_scaled = self.scaler.transform(features)
            clusters = self.kmeans.predict(features_scaled)

            for idx, cluster in zip(indices, clusters):
                results[idx]['cluster'] = int(cluster)

        # Tambahkan info compatible_with per sel
        for r in results:
            if r['cluster'] is not None:
                r['compatible_with'] = [
                    other['cell_id'] for other in results
                    if other['cell_id'] != r['cell_id'] and
                    other['cluster'] == r['cluster']
                ]
            else:
                r['compatible_with'] = []

        return results

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model disimpan ke: {path}")

    @staticmethod
    def load(path: str) -> 'CellMatcher':
        with open(path, 'rb') as f:
            return pickle.load(f)
```

### Script Training

Buat file `train.py` di root folder:

```python
from src.data_loader import load_all_cells
from src.feature_extractor import extract_snapshot_features
from src.model import CellMatcher
import pandas as pd

# 1. Load semua sel
cells = load_all_cells('data/hnei/')

# 2. Ekstrak snapshot fitur per sel (ambil di cycle ke-100 sebagai representasi awal pakai)
snapshots = []
for cell in cells:
    snap = extract_snapshot_features(cell, at_cycle=100)
    if snap:
        snapshots.append(snap)

df = pd.DataFrame(snapshots).dropna(subset=['IR_ohm', 'OCV_V'])
print(f"Total sel valid untuk training: {len(df)}")
print(df[['cell_id', 'SoH', 'IR_ohm', 'OCV_V']].to_string())

# 3. Training
model = CellMatcher(n_clusters=3)
model.fit(df)

# 4. Simpan model
model.save('models/cell_matcher_v1.pkl')
```

Jalankan:
```bash
python train.py
```

---

## 8. Evaluasi Model

Di notebook `03_modeling.ipynb`:

```python
from src.model import CellMatcher
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

model = CellMatcher.load('models/cell_matcher_v1.pkl')

# Test dengan data sel yang belum pernah dilihat model
test_cells = [
    {'cell_id': 'test_sel_1', 'IR_ohm': 0.045, 'OCV_V': 3.85},
    {'cell_id': 'test_sel_2', 'IR_ohm': 0.046, 'OCV_V': 3.84},   # harusnya kompatibel dengan sel_1
    {'cell_id': 'test_sel_3', 'IR_ohm': 0.120, 'OCV_V': 3.40},   # harusnya outlier (degraded)
    {'cell_id': 'test_sel_4', 'IR_ohm': 0.044, 'OCV_V': 3.86},
]

results = model.predict(test_cells)
for r in results:
    status = "OUTLIER (degraded)" if r['is_outlier'] else f"Cluster {r['cluster']}"
    print(f"{r['cell_id']}: {status} | compatible: {r['compatible_with']}")
```

**Yang perlu diperiksa:**
- Silhouette score > 0.5 = clustering bagus
- Sel dengan IR tinggi (>2x rata-rata) harus masuk outlier
- Sel dengan IR mirip harus masuk cluster yang sama

---

## 9. Export & Serving Model

Model disimpan sebagai `.pkl` via `model.save()`. Untuk serving, load model sekali saat startup API.

Buat file `src/predictor.py`:

```python
from src.model import CellMatcher

_model = None

def get_model() -> CellMatcher:
    global _model
    if _model is None:
        _model = CellMatcher.load('models/cell_matcher_v1.pkl')
    return _model

def predict_cells(cells: list[dict]) -> list[dict]:
    model = get_model()
    return model.predict(cells)
```

---

## 10. API Layer dengan FastAPI

Buat file `api/schema.py`:

```python
from pydantic import BaseModel

class CellInput(BaseModel):
    cell_id: str
    IR_ohm: float       # internal resistance dalam Ohm (dari pengukuran mekanik)
    OCV_V: float        # open circuit voltage dalam Volt

class CellResult(BaseModel):
    cell_id: str
    IR_ohm: float
    OCV_V: float
    is_outlier: bool            # True = sel degraded, tidak layak direpack
    cluster: int | None         # ID cluster kompatibilitas
    compatible_with: list[str]  # cell_id lain yang kompatibel

class PredictRequest(BaseModel):
    cells: list[CellInput]

class PredictResponse(BaseModel):
    results: list[CellResult]
    summary: dict
```

Buat file `api/main.py`:

```python
from fastapi import FastAPI
from api.schema import PredictRequest, PredictResponse
from src.predictor import predict_cells

app = FastAPI(
    title="ReVoltz Cell Matcher API",
    description="AI engine untuk menentukan kompatibilitas sel baterai EV bekas",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok", "service": "ReVoltz Cell Matcher"}

@app.post("/predict-cells", response_model=PredictResponse)
def predict(request: PredictRequest):
    cells_input = [c.model_dump() for c in request.cells]
    results = predict_cells(cells_input)

    n_outlier = sum(1 for r in results if r['is_outlier'])
    n_valid = len(results) - n_outlier

    return PredictResponse(
        results=results,
        summary={
            "total_cells": len(results),
            "outlier_count": n_outlier,
            "valid_for_repack": n_valid,
            "clusters_found": len(set(r['cluster'] for r in results if r['cluster'] is not None))
        }
    )

@app.get("/health")
def health():
    return {"status": "healthy"}
```

Jalankan lokal:
```bash
uvicorn api.main:app --reload --port 8000
```

Test di browser: `http://localhost:8000/docs` — Swagger UI otomatis tersedia.

Contoh request body untuk test:
```json
{
  "cells": [
    {"cell_id": "sel_A", "IR_ohm": 0.045, "OCV_V": 3.85},
    {"cell_id": "sel_B", "IR_ohm": 0.046, "OCV_V": 3.84},
    {"cell_id": "sel_C", "IR_ohm": 0.120, "OCV_V": 3.40}
  ]
}
```

---

## 11. Deploy ke Railway / Render

### Buat Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Deploy ke Railway (rekomendasi — paling mudah)

1. Push project ke GitHub
2. Buka https://railway.app → New Project → Deploy from GitHub
3. Pilih repo, Railway auto-detect Dockerfile
4. Set environment variable jika ada
5. Dapat URL publik otomatis — share ke frontend

### Deploy ke Render (alternatif)

1. Buka https://render.com → New Web Service
2. Connect GitHub repo
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

---

## 12. Catatan Teknis Penting

### IR Tidak Tersedia Langsung di HNEI BatteryLife
Field `internal_resistance_in_ohm` adalah `None` di seluruh 1086 cycle dalam format BatteryLife. IR **harus diekstrak** menggunakan metode DCIR dari voltage/current time-series (Section 5). Ini bukan kelemahan — metode DCIR adalah standar industri yang dipakai riset BMS.

### Jumlah Cluster K-Means
Default `n_clusters=3` merepresentasikan tiga grade kompatibilitas (tinggi, sedang, rendah). Tune nilai ini berdasarkan silhouette score — jalankan loop `k = 2..6` dan pilih k dengan score tertinggi.

### Input Mekanik dari Bengkel vs Data Training
Data training berasal dari pengukuran lab (HNEI). Data input bengkel berasal dari multimeter dan IR meter sederhana. Ada kemungkinan offset sistematis — pertimbangkan normalisasi input sebelum prediksi jika akurasi kurang memuaskan.

### Threshold Kompatibilitas
Threshold `delta_IR < 5mΩ` dan `delta_OCV < 50mV` diambil dari literatur BMS standar. Nilai ini bisa di-tune setelah mendapat feedback dari bengkel partner pertama.

### Roadmap Setelah MVP
1. Tambahkan Stanford Parallel Module dataset untuk reinforce clustering
2. Collect data IR aktual dari bengkel → retrain dengan ground truth nyata
3. Tambahkan chemistry sebagai fitur tambahan (LFP vs NMC punya profil IR berbeda)
4. Versioning model dengan MLflow atau DVC

---

*ReVoltz — AI-driven EV Battery Decision System*
*Model 2: Cell Matcher | Developed for IYREF 2026 Hackathon*
