
# Broiler Weight Estimation Pipeline

A modular, end-to-end pipeline for estimating broiler weight using RGB and depth images with 2D geometric features, 3D depth features, and ResNet features.

## Overview

This pipeline implements the broiler weight estimation approach described in the research paper, combining multiple feature types through gradient boosting decision trees (LightGBM and XGBoost).

### Features

- **Multi-modal input**: Supports RGB and depth image pairs
- **Feature extraction**: 2D geometric, 3D depth, and ResNet features
- **Flexible fusion**: Concatenation-based feature fusion with extensible architecture
- **Multiple models**: LightGBM and XGBoost support with CPU/GPU acceleration
- **Modular design**: Easy to extend and customize
- **CLI tools**: Command-line interfaces for training and inference
- **Reproducible**: Deterministic seed handling and configuration management

## Quick Start

### 1. Setup Environment

Install dependencies:
```bash
pip install -r requirements.txt
```

For GPU acceleration (optional):
```bash
pip install lightgbm[gpu] xgboost[gpu]
```

For ResNet feature extraction (optional):
```bash
pip install torch torchvision
```

### 2. Single Command Inference (Recommended)

**Run the complete pipeline with one command:**

```bash
python pipeline/pipeline.py --input data/test_sample/
```

This single command will:
1. Load RGB and depth images from the input directory
2. Preprocess images (resize, normalize, depth scaling)
3. Extract 2D geometric features, 3D depth features, and ResNet features
4. Combine features using fusion strategy
5. Run inference using trained models from `models/saved_models 7 55/`
6. Display predictions in console

**Advanced usage:**

```bash
# Save predictions to CSV
python pipeline/pipeline.py --input data/test_sample/ --output predictions.csv

# Use XGBoost instead of LightGBM
python pipeline/pipeline.py --input data/test_sample/ --model xgb

# Custom configuration file
python pipeline/pipeline.py --input data/test_sample/ --config my_config.yaml

# Verbose logging
python pipeline/pipeline.py --input data/test_sample/ --verbose
```

### 3. Train Models (Optional)

If you need to retrain models:
```bash
python pipeline/train.py --config pipeline/config.yaml
```

This will:
- Load the dataset from `models/dataset_pruned.csv`
- Train LightGBM and XGBoost models
- Save trained models to `models/trained_TIMESTAMP/`

### 3. Run Inference

**Single image pair:**
```bash
python pipeline/infer.py \
  --rgb-image data/rgb/sample.png \
  --depth-file data/depth/sample.npy \
  --config pipeline/config.yaml
```

**Batch inference:**
```bash
python pipeline/infer.py \
  --rgb-dir data/rgb \
  --depth-dir data/depth \
  --output predictions.csv \
  --config pipeline/config.yaml
```

### 4. Test Installation

Run smoke tests to verify installation:
```bash
python pipeline/smoke_test.py
```

## Directory Structure

```
pipeline/
├── __init__.py              # Package initialization
├── config.yaml              # Default configuration
├── pipeline.py              # Main pipeline class
├── data_loader.py           # Data loading utilities
├── feature_extractors.py    # Feature extraction modules
├── train.py                 # Training CLI script
├── infer.py                 # Inference CLI script
├── smoke_test.py            # Smoke test script
├── utils.py                 # Utility functions
└── README.md                # This file

notebooks/
└── example_inference.ipynb  # Example usage notebook
```

## Configuration

The pipeline is configured via `pipeline/config.yaml`. Key sections:

### Data Paths
```yaml
data:
  rgb_dir: "data/rgb"           # RGB image directory
  depth_dir: "data/depth"       # Depth file directory (.npy)
  dataset_csv: "models/dataset_pruned.csv"  # Training dataset
```

### Model Configuration
```yaml
model:
  artifacts_dir: "models/saved_models 7 55"  # Trained model directory
  lgbm_model: "lgbm_model.joblib"
  xgb_model: "xgb_gpu_model.joblib"
  scaler: "scaler.joblib"
  imputer: "imputer.joblib"
```

### Preprocessing
```yaml
preprocessing:
  rgb:
    resize: [224, 224]
    normalize_mean: [0.485, 0.456, 0.406]  # ImageNet normalization
    normalize_std: [0.229, 0.224, 0.225]
  depth:
    scale_factor: 1.0
    fill_missing: 0.0
    max_depth: 2000.0
```

## Single Entry Point Pipeline

The pipeline can be executed with a single command that runs all steps automatically:

```bash
python pipeline/pipeline.py --input <input_directory>
```

### Pipeline Flow

The single entry point executes these steps sequentially:

1. **Data Loading**: Discovers and loads RGB-Depth image pairs
2. **Preprocessing**: Resizes, normalizes, and scales images
3. **Feature Extraction**:
   - 2D geometric features (area, perimeter, compactness, etc.)
   - 3D depth features (volume, surface area, height statistics)
   - ResNet features (2048-dimensional CNN features)
4. **Feature Fusion**: Concatenates all features into single vector
5. **Inference**: Predicts broiler weight using trained GBDT models
6. **Output**: Displays results and optionally saves to CSV

### Command Line Arguments

- `--input`: **Required.** Directory containing `rgb/` and `depth/` subdirectories
- `--config`: Configuration file path (default: `pipeline/config.yaml`)
- `--output`: Output CSV file for predictions (optional)
- `--model`: Model type - `lgbm` or `xgb` (default: `lgbm`)
- `--verbose`: Enable detailed logging

### Examples

```bash
# Basic inference
python pipeline/pipeline.py --input data/my_broilers/

# Save results to CSV
python pipeline/pipeline.py --input data/my_broilers/ --output results.csv

# Use XGBoost model with verbose output
python pipeline/pipeline.py --input data/my_broilers/ --model xgb --verbose
```

### Expected Output

```
============================================================
Starting Broiler Weight Estimation Pipeline
============================================================
Input directory: data/test_sample/
RGB directory: data/test_sample/rgb
Depth directory: data/test_sample/depth
Model type: lgbm

========================================
STEP 1: Initializing Pipeline
========================================
INFO - Initialized 2D feature extractor with 25 features
INFO - Initialized 3D feature extractor
INFO - Initialized ResNet feature extractor: resnet50

========================================
STEP 2: Loading Models
========================================
INFO - Loading models from: models/saved_models 7 55
INFO - Loaded LightGBM model with MAE: 0.089

========================================
STEP 3: Loading and Preprocessing Data
========================================
INFO - Found 5 RGB-Depth image pairs

========================================
STEP 4: Feature Extraction and Inference
========================================
INFO - Processing sample 1/5: rgb_sample_001.png
INFO - Predicted weight: 2.456 kg
INFO - Processing sample 2/5: rgb_sample_002.png
INFO - Predicted weight: 3.123 kg
...

========================================
STEP 5: Results Summary
========================================
Processed 5 samples
Mean predicted weight: 2.890 kg
Min predicted weight: 2.456 kg
Max predicted weight: 3.456 kg

Individual Predictions:
------------------------------------------------------------
  1. rgb_sample_001.png                    -> 2.456 kg
  2. rgb_sample_002.png                    -> 3.123 kg
  3. rgb_sample_003.png                    -> 2.789 kg
  4. rgb_sample_004.png                    -> 3.456 kg
  5. rgb_sample_005.png                    -> 2.626 kg

============================================================
Pipeline execution completed successfully!
============================================================
```

## Data Format

### Expected File Structure
```
data/
├── rgb/
│   ├── rgb_20250725_162941_594732_instance-0.png
│   ├── rgb_20250725_162941_594732_instance-1.png
│   └── ...
└── depth/
    ├── depth_20250725_162941_594732_instance-0.npy
    ├── depth_20250725_162941_594732_instance-1.npy
    └── ...
```

### File Naming Convention
- RGB images: `rgb_TIMESTAMP_instance-ID.png`
- Depth files: `depth_TIMESTAMP_instance-ID.npy`
- Timestamps and instance IDs must match between RGB and depth files

### Data Requirements
- **RGB images**: PNG/JPG format, any resolution (will be resized)
- **Depth files**: NumPy arrays (.npy), same spatial dimensions as RGB
- **Depth values**: Millimeters or consistent depth units
- **Missing values**: Use 0 or NaN for background/missing depth

## API Usage

### Pipeline Class

```python
from pipeline.pipeline import BroilerWeightPipeline

# Initialize pipeline
pipeline = BroilerWeightPipeline("pipeline/config.yaml")

# Load trained models
pipeline.load_models()

# Predict single sample
weight = pipeline.predict_from_images("rgb.png", "depth.npy")

# Predict batch
results = pipeline.predict_from_directory("rgb_dir", "depth_dir")
```

### Feature Extraction

```python
from pipeline.feature_extractors import FeatureExtractor2D, FeatureExtractor3D

# Extract 2D features
extractor_2d = FeatureExtractor2D(config)
features_2d = extractor_2d.extract(rgb_image, rgb_path)

# Extract 3D features  
extractor_3d = FeatureExtractor3D(config)
features_3d = extractor_3d.extract(depth_data, depth_path)
```

## Training Details

### Dataset Requirements
The training dataset should be a CSV file with:
- `weight_kg`: Target weight labels
- Feature columns: Pre-extracted features (2D + 3D + ResNet)
- ID columns: Sample identifiers (optional, excluded from training)

### Model Training Process
1. Load dataset from CSV
2. Split into train/test sets (70/30 default)
3. Apply median imputation for missing values
4. Apply standard scaling for normalization
5. Train LightGBM and XGBoost models
6. Evaluate on test set (MAE, RMSE, R²)
7. Save models and metrics

### Training Output
```
models/trained_TIMESTAMP/
├── imputer.joblib          # Missing value imputer
├── scaler.joblib           # Feature scaler
├── lgbm_model.joblib       # LightGBM model
├── xgb_gpu_model.joblib    # XGBoost model
├── metrics.json            # Evaluation metrics
└── training_details.txt    # Training configuration
```

## Performance and Hardware

### CPU vs GPU
- **CPU**: All functionality works on CPU
- **GPU**: Optional acceleration for XGBoost/LightGBM training
- **Memory**: ~2GB RAM recommended for typical datasets
- **Storage**: ~100MB for trained models

### Batch Processing
- Default batch size: 32 samples
- Memory usage scales with batch size and image resolution
- Adjust `batch_size` in config for memory constraints

## Troubleshooting

### Common Issues

**Models not found:**
```
RuntimeError: Models not loaded. Call load_models() first.
```
Solution: Train models first with `python pipeline/train.py`

**Image pair mismatch:**
```
ValueError: Invalid image pair: RGB file not found
```
Solution: Verify file naming convention and directory structure

**Feature dimension mismatch:**
```
ValueError: Feature dimension mismatch
```
Solution: Ensure consistent feature extraction settings between training and inference

**Memory errors:**
```
MemoryError: Unable to allocate array
```
Solution: Reduce batch size in config or process smaller batches

### Debug Mode
Enable verbose logging:
```bash
python pipeline/infer.py --verbose ...
```

### Validation
Run smoke tests to validate installation:
```bash
python pipeline/smoke_test.py
```

## Extending the Pipeline

### Adding New Feature Extractors
1. Create new extractor class in `feature_extractors.py`
2. Implement `extract()` method with consistent interface
3. Update `FeatureFusion` to include new features
4. Update configuration to enable/disable features

### Adding New Models
1. Install model library
2. Add training code in `train.py`
3. Add prediction code in `pipeline.py`
4. Update configuration with new model settings

### Custom Preprocessing
1. Modify `data_loader.py` for custom image preprocessing
2. Update configuration with new preprocessing parameters
3. Ensure consistency between training and inference

## Contributing

1. Follow existing code style and documentation patterns
2. Add unit tests for new functionality
3. Update this README for new features
4. Test with smoke test script

## License

This project is part of the flocksense-ai broiler weight estimation research.
