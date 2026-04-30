# Notebooks

No `.ipynb` files were required for the first implementation because the MVP was built as a runnable training and serving pipeline first.

That was intentional:
- `train.py` is better for repeatable training
- FastAPI integration needs script-first code anyway
- notebooks are better for exploration, not for the final serving path

If you want, the next step can be to add:
- `01_eda.ipynb`
- `02_feature_engineering.ipynb`
- `03_modeling.ipynb`

Those notebooks would be useful for:
- visualizing capacity fade and SoH trends
- inspecting IR extraction behavior from raw cycles
- comparing clustering quality across different `k` values

