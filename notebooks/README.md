# Notebooks
Notebook ini mencakup alur end-to-end:
- eksplorasi dataset HNEI (ringkasan cell + cycle)
- EDA SoH/capacity fade
- analisis feature engineering (`IR_ohm`, `OCV_V`, `SoH`)
- training dan evaluasi `CellMatcher` (outlier, cluster, silhouette metadata)
- inferensi dan rekomendasi pack

Notebook Model 1 (`soh_predictor.ipynb`) mencakup:
- ingestion NASA ARC zip dataset
- EDA battery-level dan cycle-level
- training/evaluasi SoH predictor
- inferensi pack-level + action triage

## Cara pakai

1. Pastikan dependency sudah ter-install:
   - `pip install -r requirements.txt`
2. Jalankan Jupyter:
   - `jupyter notebook`
3. Buka:
   - `notebooks/cell_matcher.ipynb`
4. Run semua sel dari atas ke bawah.



