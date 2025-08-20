
#!/usr/bin/env python3
"""
Training script for broiler weight estimation pipeline.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import json
from datetime import datetime

# Add pipeline to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pipeline import BroilerWeightPipeline
from pipeline.data_loader import load_training_dataset
from pipeline.utils import load_config, ensure_directory, setup_logging

logger = setup_logging()


def train_models(config_path: str, output_dir: str = None) -> None:
    """
    Train broiler weight estimation models.
    
    Args:
        config_path: Path to configuration file
        output_dir: Output directory for models (optional)
    """
    # Load configuration
    config = load_config(config_path)
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"models/trained_{timestamp}"
    
    ensure_directory(output_dir)
    logger.info(f"Training models, output directory: {output_dir}")
    
    # Load dataset
    dataset_path = config['data']['dataset_csv']
    logger.info(f"Loading dataset from {dataset_path}")
    df = load_training_dataset(dataset_path)
    
    # Prepare features and labels
    # Assuming the dataset has 'weight_kg' column and feature columns
    if 'weight_kg' not in df.columns:
        raise ValueError("Dataset must contain 'weight_kg' column")
    
    # Separate features and target
    target_col = 'weight_kg'
    feature_cols = [col for col in df.columns if col != target_col and not col.startswith('id')]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    logger.info(f"Dataset shape: {X.shape}, target shape: {y.shape}")
    
    # Train-test split
    test_size = config['training']['test_size']
    random_state = config['training']['random_state']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Preprocessing
    logger.info("Fitting preprocessors...")
    
    # Imputer for missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scaler for normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Train models
    models = {}
    
    # LightGBM
    try:
        import lightgbm as lgb
        logger.info("Training LightGBM model...")
        
        lgbm_model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=10,
            random_state=random_state,
            verbose=-1
        )
        lgbm_model.fit(X_train_scaled, y_train)
        models['lgbm'] = lgbm_model
        
    except ImportError:
        logger.warning("LightGBM not available, skipping")
    
    # XGBoost
    try:
        import xgboost as xgb
        logger.info("Training XGBoost model...")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=10,
            random_state=random_state,
            tree_method='hist'  # Use CPU fallback
        )
        xgb_model.fit(X_train_scaled, y_train)
        models['xgb'] = xgb_model
        
    except ImportError:
        logger.warning("XGBoost not available, skipping")
    
    if not models:
        raise RuntimeError("No models could be trained. Install lightgbm or xgboost.")
    
    # Evaluate models
    logger.info("Evaluating models...")
    metrics = {}
    
    for model_name, model in models.items():
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        train_metrics = {
            'mae': mean_absolute_error(y_train, y_pred_train),
            'mse': mean_squared_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'r2': r2_score(y_train, y_pred_train)
        }
        
        test_metrics = {
            'mae': mean_absolute_error(y_test, y_pred_test),
            'mse': mean_squared_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'r2': r2_score(y_test, y_pred_test)
        }
        
        metrics[model_name] = {
            'train': train_metrics,
            'test': test_metrics
        }
        
        logger.info(f"{model_name.upper()} - Test RMSE: {test_metrics['rmse']:.4f}, "
                   f"Test MAE: {test_metrics['mae']:.4f}, Test R2: {test_metrics['r2']:.4f}")
    
    # Save models and preprocessors
    logger.info(f"Saving models to {output_dir}...")
    
    # Save preprocessors
    joblib.dump(imputer, os.path.join(output_dir, "imputer.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    
    # Save models
    for model_name, model in models.items():
        model_filename = f"{model_name}_model.joblib"
        if model_name == 'xgb':
            model_filename = "xgb_gpu_model.joblib"  # Match expected naming
        joblib.dump(model, os.path.join(output_dir, model_filename))
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save training details
    training_details = {
        'timestamp': datetime.now().isoformat(),
        'dataset_path': dataset_path,
        'dataset_shape': list(df.shape),
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'feature_count': X_train.shape[1],
        'test_size': test_size,
        'random_state': random_state,
        'models_trained': list(models.keys())
    }
    
    with open(os.path.join(output_dir, "training_details.txt"), 'w') as f:
        for key, value in training_details.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Training completed! Models saved to {output_dir}")


def main():
    """Main function for training CLI."""
    parser = argparse.ArgumentParser(description="Train broiler weight estimation models")
    parser.add_argument("--config", default="pipeline/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output-dir", 
                       help="Output directory for trained models")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        train_models(args.config, args.output_dir)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
