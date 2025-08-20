
"""
Utility functions for the broiler weight estimation pipeline.
"""

import os
import yaml
import numpy as np
import random
import joblib
import json
from typing import Dict, Any, Tuple, Optional
import logging


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "pipeline/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # Note: torch.manual_seed(seed) would go here if using PyTorch


def validate_file_exists(file_path: str) -> None:
    """Validate that a file exists, raise FileNotFoundError if not."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")


def validate_image_pair(rgb_path: str, depth_path: str) -> Tuple[bool, str]:
    """
    Validate that RGB and depth image pair exists and are readable.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        validate_file_exists(rgb_path)
        validate_file_exists(depth_path)
        
        # Basic format validation
        if not rgb_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return False, f"RGB file must be PNG/JPG: {rgb_path}"
        if not depth_path.lower().endswith('.npy'):
            return False, f"Depth file must be NPY: {depth_path}"
            
        return True, ""
    except FileNotFoundError as e:
        return False, str(e)


def load_model_artifacts(artifacts_dir: str) -> Dict[str, Any]:
    """
    Load all model artifacts from the specified directory.
    
    Returns:
        Dictionary containing loaded models and preprocessors
    """
    artifacts = {}
    
    # Load scaler and imputer
    scaler_path = os.path.join(artifacts_dir, "scaler.joblib")
    imputer_path = os.path.join(artifacts_dir, "imputer.joblib")
    
    validate_file_exists(scaler_path)
    validate_file_exists(imputer_path)
    
    artifacts['scaler'] = joblib.load(scaler_path)
    artifacts['imputer'] = joblib.load(imputer_path)
    
    # Load models
    lgbm_path = os.path.join(artifacts_dir, "lgbm_model.joblib")
    xgb_path = os.path.join(artifacts_dir, "xgb_gpu_model.joblib")
    
    if os.path.exists(lgbm_path):
        artifacts['lgbm_model'] = joblib.load(lgbm_path)
    if os.path.exists(xgb_path):
        artifacts['xgb_model'] = joblib.load(xgb_path)
    
    # Load metrics if available
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            artifacts['metrics'] = json.load(f)
    
    return artifacts


def ensure_directory(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def get_matching_depth_path(rgb_path: str, depth_dir: str) -> Optional[str]:
    """
    Get corresponding depth file path for an RGB image.
    
    Assumes RGB files are named like: rgb_timestamp_instance-X.png
    And depth files are named like: depth_timestamp_instance-X.npy
    """
    rgb_filename = os.path.basename(rgb_path)
    if rgb_filename.startswith('rgb_'):
        depth_filename = rgb_filename.replace('rgb_', 'depth_raw_').replace('.png', '.npy').replace('.jpg', '.npy')
        depth_path = os.path.join(depth_dir, depth_filename)
        if os.path.exists(depth_path):
            return depth_path
    return None
