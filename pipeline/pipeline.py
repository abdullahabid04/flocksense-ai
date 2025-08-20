"""
Main pipeline class for broiler weight estimation.
"""
import pprint

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import os

from .data_loader import ImagePairDataLoader, load_training_dataset
from .feature_extractors import FeatureExtractor2D, FeatureExtractor3D, FeatureExtractorResNet, FeatureFusion
from .utils import load_config, load_model_artifacts, set_random_seeds, setup_logging

logger = setup_logging()


class BroilerWeightPipeline:
    """
    End-to-end pipeline for broiler weight estimation.

    Handles data loading, preprocessing, feature extraction, fusion, and prediction.
    """

    def __init__(self, config_path: str = "pipeline/config.yaml"):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        set_random_seeds(self.config['training']['random_state'])

        # Initialize feature extractors
        self.extractor_2d = FeatureExtractor2D(self.config)
        self.extractor_3d = FeatureExtractor3D(self.config)
        self.extractor_resnet = FeatureExtractorResNet(self.config)
        self.feature_fusion = FeatureFusion(self.config)

        # Model artifacts (loaded when needed)
        self.artifacts = None
        self.is_trained = False

    def load_models(self) -> None:
        """Load trained models and preprocessors."""
        artifacts_dir = self.config['model']['artifacts_dir']
        self.artifacts = load_model_artifacts(artifacts_dir)
        self.is_trained = True
        logger.info(f"Loaded models from {artifacts_dir}")

    def extract_features_from_pair(self, rgb_image: np.ndarray, depth_data: np.ndarray,
                                   rgb_path: str = None, depth_path: str = None) -> np.ndarray:
        """
        Extract and fuse features from RGB-Depth pair.

        Args:
            rgb_image: RGB image as numpy array
            depth_data: Depth data as numpy array
            rgb_path: Optional RGB file path
            depth_path: Optional depth file path

        Returns:
            Fused feature vector
        """
        # Extract features from each modality
        features_2d = self.extractor_2d.extract(rgb_image, rgb_path)
        features_3d = self.extractor_3d.extract(depth_data, depth_path)
        features_resnet = self.extractor_resnet.extract(rgb_image)

        # Fuse features
        fused_features = self.feature_fusion.fuse(features_2d, features_3d, features_resnet)

        logger.debug(f"2D features: {len(features_2d)}")
        logger.debug(f"3D features: {len(features_3d)}")
        logger.debug(f"ResNet features: {features_resnet.shape}")
        logger.debug(f"Fused features: {fused_features.shape}")

        return fused_features

    def extract_features_batch(self, data_loader: ImagePairDataLoader) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from a batch of RGB-Depth pairs.

        Args:
            data_loader: ImagePairDataLoader instance

        Returns:
            Tuple of (feature_matrix, sample_ids)
        """
        all_features = []
        sample_ids = []

        batch_size = self.config['training']['batch_size']

        for batch in data_loader.batch_generator(batch_size):
            rgb_batch, depth_batch, rgb_paths, depth_paths = batch

            for i in range(len(rgb_batch)):
                rgb_image = rgb_batch[i]
                depth_data = depth_batch[i]
                rgb_path = rgb_paths[i]
                depth_path = depth_paths[i]

                try:
                    features = self.extract_features_from_pair(
                        rgb_image, depth_data, rgb_path, depth_path)
                    all_features.append(features)
                    sample_ids.append(os.path.basename(rgb_path))

                except Exception as e:
                    logger.warning(f"Failed to extract features for {rgb_path}: {e}")
                    continue

        if not all_features:
            raise ValueError("No features were successfully extracted")

        feature_matrix = np.vstack(all_features)
        logger.info(f"Extracted features for {len(all_features)} samples, "
                    f"feature dimension: {feature_matrix.shape[1]}")

        return feature_matrix, sample_ids

    def preprocess_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Preprocess features using imputation and scaling.

        Args:
            features: Feature matrix
            fit: Whether to fit the preprocessors (training mode)

        Returns:
            Preprocessed feature matrix
        """
        if not self.is_trained:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Impute missing values
        features_imputed = self.artifacts['imputer'].transform(features)

        # Scale features
        features_scaled = self.artifacts['scaler'].transform(features_imputed)

        return features_scaled

    def predict(self, features: np.ndarray, model_type: str = None) -> float:
        """
        Make weight predictions using the specified model.

        Args:
            features: Preprocessed feature matrix
            model_type: Model type ('lgbm' or 'xgb'). Uses config default if None.

        Returns:
            Predicted weights
        """
        if not self.is_trained:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        if model_type is None:
            model_type = self.config['inference']['model_type']

        # Ensure features is 2D (1 sample, N features)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        model_key = f'{model_type}_model'
        if model_key not in self.artifacts:
            raise ValueError(f"Model {model_type} not available. "
                             f"Available: {[k for k in self.artifacts.keys() if k.endswith('_model')]}")

        model = self.artifacts[model_key]

        predictions = model.predict(features)

        prediction = float(np.asarray(predictions).reshape(-1)[0])

        return prediction

    def predict_from_images(self, rgb_path: str, depth_path: str,
                            model_type: str = None) -> float:
        """
        Predict weight from a single RGB-Depth image pair.

        Args:
            rgb_path: Path to RGB image
            depth_path: Path to depth file
            model_type: Model type to use

        Returns:
            Predicted weight in kg
        """
        if not self.is_trained:
            self.load_models()

        # Create temporary data loader for single pair
        rgb_dir = os.path.dirname(rgb_path)
        depth_dir = os.path.dirname(depth_path)

        data_loader = ImagePairDataLoader(rgb_dir, depth_dir, self.config)

        # Load and preprocess the specific pair
        rgb_image = data_loader.load_rgb_image(rgb_path)
        depth_data = data_loader.load_depth_data(depth_path)

        # Extract features
        features = self.extract_features_from_pair(rgb_image, depth_data, rgb_path, depth_path)
        features = features.reshape(1, -1)  # Add batch dimension

        # Preprocess and predict
        features_preprocessed = self.preprocess_features(features)
        prediction = self.predict(features_preprocessed, model_type)

        return float(prediction)

    def predict_from_directory(self, rgb_dir: str, depth_dir: str,
                               model_type: str = None) -> pd.DataFrame:
        """
        Predict weights for all RGB-Depth pairs in directories.

        Args:
            rgb_dir: Directory containing RGB images
            depth_dir: Directory containing depth files
            model_type: Model type to use

        Returns:
            DataFrame with predictions
        """
        if not self.is_trained:
            self.load_models()

        # Create data loader
        data_loader = ImagePairDataLoader(rgb_dir, depth_dir, self.config)

        if len(data_loader) == 0:
            raise ValueError(f"No valid image pairs found in {rgb_dir} and {depth_dir}")

        # Extract features
        features, sample_ids = self.extract_features_batch(data_loader)

        # Preprocess and predict
        features_preprocessed = self.preprocess_features(features)
        predictions = self.predict(features_preprocessed, model_type)

        # Create results DataFrame
        results = pd.DataFrame({
            'sample_id': sample_ids,
            'predicted_weight_kg': predictions
        })

        return results

    def get_model_info(self) -> Dict:
        """
        Get information about loaded models.

        Returns:
            Dictionary with model information
        """
        if not self.is_trained:
            return {"status": "Models not loaded"}

        info = {
            "status": "Models loaded",
            "available_models": [k for k in self.artifacts.keys() if k.endswith('_model')],
            "feature_dimensions": self.feature_fusion.get_feature_dimensions()
        }

        if 'metrics' in self.artifacts:
            info["metrics"] = self.artifacts['metrics']

        return info


def run_complete_pipeline(input_dir: str, config_path: str, output_file: Optional[str] = None,
                          model_type: str = "lgbm"):
    """Run the complete broiler weight estimation pipeline.
    
    Args:
        input_dir: Directory containing rgb/ and depth/ subdirectories
        config_path: Path to configuration file
        output_file: Optional output CSV file for predictions
        model_type: Model type to use ("lgbm" or "xgb")
    """
    logger.info("=" * 60)
    logger.info("Starting Broiler Weight Estimation Pipeline")
    logger.info("=" * 60)

    # Validate input directory structure
    rgb_dir = os.path.join(input_dir, "rgb")
    depth_dir = os.path.join(input_dir, "depth")

    if not os.path.exists(rgb_dir):
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not os.path.exists(depth_dir):
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"RGB directory: {rgb_dir}")
    logger.info(f"Depth directory: {depth_dir}")
    logger.info(f"Model type: {model_type}")

    # Step 1: Initialize pipeline
    logger.info("\n" + "=" * 40)
    logger.info("STEP 1: Initializing Pipeline")
    logger.info("=" * 40)

    pipeline = BroilerWeightPipeline(config_path=config_path)

    # Step 2: Load models
    logger.info("\n" + "=" * 40)
    logger.info("STEP 2: Loading Models")
    logger.info("=" * 40)

    pipeline.load_models()
    model_info = pipeline.get_model_info()
    logger.info(f"Loaded models: {list(model_info.keys())}")

    # Step 3: Load and preprocess data
    logger.info("\n" + "=" * 40)
    logger.info("STEP 3: Loading and Preprocessing Data")
    logger.info("=" * 40)

    data_loader = ImagePairDataLoader(rgb_dir, depth_dir, pipeline.config)
    num_samples = len(data_loader)
    logger.info(f"Found {num_samples} RGB-Depth image pairs")

    if num_samples == 0:
        logger.warning("No matching RGB-Depth pairs found!")
        return

    # Step 4: Extract features and run inference
    logger.info("\n" + "=" * 40)
    logger.info("STEP 4: Feature Extraction and Inference")
    logger.info("=" * 40)

    predictions = []
    image_paths = []

    for i in range(num_samples):
        rgb_image, depth_data, rgb_path, depth_path = data_loader[i]
        logger.info(f"Processing sample {i + 1}/{num_samples}: {os.path.basename(rgb_path)}")

        # Extract features
        features = pipeline.extract_features_from_pair(rgb_image, depth_data, rgb_path, depth_path)
        logger.debug(f"Extracted {len(features)} features")

        # Run inference
        prediction = pipeline.predict(features, model_type=model_type)

        predictions.append(prediction)
        image_paths.append(rgb_path)
        logger.info(f"Predicted weight: {prediction:.3f} kg")

    # Step 5: Output results
    logger.info("\n" + "=" * 40)
    logger.info("STEP 5: Results Summary")
    logger.info("=" * 40)

    predictions = np.array(predictions)

    logger.info(f"Processed {len(predictions)} samples")
    logger.info(f"Mean predicted weight: {predictions.mean():.3f} kg")
    logger.info(f"Min predicted weight: {predictions.min():.3f} kg")
    logger.info(f"Max predicted weight: {predictions.max():.3f} kg")
    logger.info(f"Std predicted weight: {predictions.std():.3f} kg")

    # Print individual results
    logger.info("\nIndividual Predictions:")
    logger.info("-" * 60)
    for i, (path, pred) in enumerate(zip(image_paths, predictions)):
        filename = os.path.basename(path)
        logger.info(f"{i + 1:3d}. {filename:<40} -> {pred:.3f} kg")

    # Save to CSV if requested
    if output_file:
        logger.info(f"\nSaving predictions to: {output_file}")
        import pandas as pd

        results_df = pd.DataFrame({
            'image_path': [os.path.basename(p) for p in image_paths],
            'full_path': image_paths,
            'predicted_weight_kg': predictions,
            'model_type': model_type
        })

        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline execution completed successfully!")
    logger.info("=" * 60)


def main():
    """Main entry point for the broiler weight estimation pipeline."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Broiler Weight Estimation Pipeline - Single Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference on a directory of RGB/Depth images
  python pipeline/pipeline.py --input data/test_sample/

  # Run with custom configuration
  python pipeline/pipeline.py --input data/test_sample/ --config custom_config.yaml

  # Save predictions to CSV
  python pipeline/pipeline.py --input data/test_sample/ --output predictions.csv

  # Use XGBoost instead of LightGBM
  python pipeline/pipeline.py --input data/test_sample/ --model xgb
        """
    )

    parser.add_argument("--input", required=True,
                        help="Input directory containing rgb/ and depth/ subdirectories")
    parser.add_argument("--config", default="pipeline/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output",
                        help="Output CSV file for predictions (optional)")
    parser.add_argument("--model", choices=["lgbm", "xgb"], default="lgbm",
                        help="Model type to use for inference")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        run_complete_pipeline(args.input, args.config, args.output, args.model)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
