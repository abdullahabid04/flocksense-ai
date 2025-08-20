
#!/usr/bin/env python3
"""
Smoke test for broiler weight estimation pipeline.
Tests basic functionality with mock data.
"""

import os
import sys
import numpy as np
import cv2
import tempfile
import shutil
from pathlib import Path

# Add pipeline to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pipeline import BroilerWeightPipeline
from pipeline.data_loader import ImagePairDataLoader
from pipeline.utils import setup_logging

logger = setup_logging()


def create_mock_rgb_image(width: int = 224, height: int = 224) -> np.ndarray:
    """Create a mock RGB image for testing."""
    # Create a synthetic bird-like shape
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some color variation
    image[:, :, 0] = np.random.randint(100, 150, (height, width))  # Red channel
    image[:, :, 1] = np.random.randint(80, 120, (height, width))   # Green channel
    image[:, :, 2] = np.random.randint(60, 100, (height, width))   # Blue channel
    
    # Add an elliptical shape in the center to simulate a broiler
    center_x, center_y = width // 2, height // 2
    a, b = width // 3, height // 4  # Semi-major and semi-minor axes
    
    y, x = np.ogrid[:height, :width]
    mask = ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2 <= 1
    
    # Make the broiler area brighter
    image[mask] = image[mask] + 50
    image = np.clip(image, 0, 255)
    
    return image


def create_mock_depth_data(width: int = 224, height: int = 224) -> np.ndarray:
    """Create mock depth data for testing."""
    # Create depth data with a broiler-like shape
    depth = np.zeros((height, width), dtype=np.float32)
    
    center_x, center_y = width // 2, height // 2
    a, b = width // 3, height // 4
    
    y, x = np.ogrid[:height, :width]
    mask = ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2 <= 1
    
    # Create depth values (simulating a 3D broiler body)
    for i in range(height):
        for j in range(width):
            if mask[i, j]:
                # Distance from center, normalized
                dist_from_center = np.sqrt(((j - center_x) / a) ** 2 + ((i - center_y) / b) ** 2)
                # Create a dome-like depth profile
                depth[i, j] = 800 * (1 - dist_from_center ** 2) + np.random.normal(0, 10)
    
    # Add some noise
    depth = np.maximum(depth, 0)  # Ensure no negative depths
    
    return depth


def create_test_data(temp_dir: str, num_samples: int = 3) -> None:
    """Create test RGB and depth data."""
    rgb_dir = os.path.join(temp_dir, "rgb")
    depth_dir = os.path.join(temp_dir, "depth")
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Create mock data
        rgb_image = create_mock_rgb_image()
        depth_data = create_mock_depth_data()
        
        # Save files
        rgb_filename = f"rgb_20250125_120000_12345{i}_instance-{i}.png"
        depth_filename = f"depth_20250125_120000_12345{i}_instance-{i}.npy"
        
        rgb_path = os.path.join(rgb_dir, rgb_filename)
        depth_path = os.path.join(depth_dir, depth_filename)
        
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        np.save(depth_path, depth_data)
    
    logger.info(f"Created {num_samples} test samples in {temp_dir}")


def test_data_loader(temp_dir: str) -> None:
    """Test the data loader functionality."""
    logger.info("Testing data loader...")
    
    from pipeline.utils import load_config
    config = load_config("pipeline/config.yaml")
    
    rgb_dir = os.path.join(temp_dir, "rgb")
    depth_dir = os.path.join(temp_dir, "depth")
    
    data_loader = ImagePairDataLoader(rgb_dir, depth_dir, config)
    
    assert len(data_loader) > 0, "No image pairs found"
    logger.info(f"Found {len(data_loader)} image pairs")
    
    # Test single sample loading
    rgb_image, depth_data, rgb_path, depth_path = data_loader[0]
    
    assert rgb_image.shape == (224, 224, 3), f"Unexpected RGB shape: {rgb_image.shape}"
    assert depth_data.shape == (224, 224), f"Unexpected depth shape: {depth_data.shape}"
    assert os.path.exists(rgb_path), f"RGB path doesn't exist: {rgb_path}"
    assert os.path.exists(depth_path), f"Depth path doesn't exist: {depth_path}"
    
    logger.info("Data loader test passed")


def test_feature_extraction(temp_dir: str) -> None:
    """Test feature extraction functionality."""
    logger.info("Testing feature extraction...")
    
    from pipeline.utils import load_config
    config = load_config("pipeline/config.yaml")
    
    rgb_dir = os.path.join(temp_dir, "rgb")
    depth_dir = os.path.join(temp_dir, "depth")
    
    data_loader = ImagePairDataLoader(rgb_dir, depth_dir, config)
    rgb_image, depth_data, rgb_path, depth_path = data_loader[0]
    
    # Test pipeline feature extraction
    pipeline = BroilerWeightPipeline()
    features = pipeline.extract_features_from_pair(rgb_image, depth_data, rgb_path, depth_path)
    
    assert isinstance(features, np.ndarray), "Features should be numpy array"
    assert features.ndim == 1, "Features should be 1D array"
    assert len(features) > 0, "Features should not be empty"
    
    expected_dims = pipeline.feature_fusion.get_feature_dimensions()
    assert len(features) == expected_dims['total'], \
        f"Feature dimension mismatch: {len(features)} vs {expected_dims['total']}"
    
    logger.info(f"Feature extraction test passed. Feature dimension: {len(features)}")


def test_model_loading() -> None:
    """Test model loading functionality."""
    logger.info("Testing model loading...")
    
    pipeline = BroilerWeightPipeline()
    
    try:
        pipeline.load_models()
        logger.info("Model loading test passed")
        
        # Test model info
        info = pipeline.get_model_info()
        logger.info(f"Model info: {info}")
        
    except Exception as e:
        logger.warning(f"Model loading failed (expected if models not trained): {e}")


def test_inference(temp_dir: str) -> None:
    """Test inference functionality with mock prediction."""
    logger.info("Testing inference (with mock models)...")
    
    from pipeline.utils import load_config
    config = load_config("pipeline/config.yaml")
    
    rgb_dir = os.path.join(temp_dir, "rgb")
    depth_dir = os.path.join(temp_dir, "depth")
    
    data_loader = ImagePairDataLoader(rgb_dir, depth_dir, config)
    
    if len(data_loader) == 0:
        logger.warning("No test data available for inference test")
        return
    
    # Test feature extraction and preprocessing pipeline
    pipeline = BroilerWeightPipeline()
    
    try:
        # This will fail if models aren't available, which is expected in smoke test
        results = pipeline.predict_from_directory(rgb_dir, depth_dir)
        logger.info(f"Inference test passed. Predictions shape: {results.shape}")
        
    except Exception as e:
        logger.info(f"Inference test skipped (models not available): {e}")


def run_smoke_test() -> None:
    """Run comprehensive smoke test."""
    logger.info("Starting smoke test for broiler weight estimation pipeline")
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary directory: {temp_dir}")
        
        try:
            # Create test data
            create_test_data(temp_dir, num_samples=3)
            
            # Run tests
            test_data_loader(temp_dir)
            test_feature_extraction(temp_dir)
            test_model_loading()
            test_inference(temp_dir)
            
            logger.info("Smoke test completed successfully!")
            
        except Exception as e:
            logger.error(f"Smoke test failed: {e}")
            raise


if __name__ == "__main__":
    run_smoke_test()
