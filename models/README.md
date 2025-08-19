# Broiler Weight Estimation with GBDT (GPU-ready)

This project trains Gradient Boosting Decision Trees (LightGBM, XGBoost) on fused 2D/3D/C-ResNet50 features to estimate broiler weight, following the referenced paper.

## Setup

1) Create a Python environment and install dependencies:

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

(Optional GPU builds)
- XGBoost: wheels include GPU (use `tree_method=gpu_hist`).
- LightGBM: needs GPU-enabled build; this notebook falls back to CPU if not available.

2) Register a Jupyter kernel (optional):
```
python -m ipykernel install --user --name broiler-gbdt --display-name "Python (broiler-gbdt)"
```

## Data
- Input: `dataset_pruned.csv` (IDs, label `weight_kg`, 25 artificial features, and 2048 learned features).

## Notebooks
- `GBDT_3090.ipynb`: GPU-accelerated training (RTX 3090). Performs:
  - 70/30 split, feature normalization, median imputation
  - LightGBM (GPU if available) and XGBoost (gpu_hist)
  - Metrics: MAE, MSE, RMSE, R2
  - Saves artifacts to `saved models/` and `artifacts_3090/`

## Outputs
- `saved models/`
  - `imputer.joblib`, `scaler.joblib`
  - `lgbm_model.joblib`, `xgb_gpu_model.joblib`
  - `metrics.json`, `training_3090_details.txt`

## Repro tips
- Ensure NVIDIA drivers and CUDA-compatible toolkit for best XGBoost GPU performance.
- If LightGBM GPU is not present, it will auto-fallback to CPU.
