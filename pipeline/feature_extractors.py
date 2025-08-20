"""
Feature extraction modules for 2D, 3D, and ResNet features.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
import torch
from torchvision import transforms
from PIL import Image
import logging

# Add root directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract_all_features import extract_2d_features_from_image, extract_3d_features_from_depth
from .utils import setup_logging

logger = setup_logging()


class FeatureExtractor2D:
    """Extract 2D geometric features from RGB images."""

    def __init__(self, config: Dict):
        """
        Initialize 2D feature extractor.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def extract(self, rgb_image: np.ndarray, rgb_path: str = None) -> Dict[str, float]:
        """
        Extract 2D features from RGB image.

        Args:
            rgb_image: RGB image as numpy array
            rgb_path: Optional path to save/load from (for compatibility)

        Returns:
            Dictionary of 2D features
        """
        try:
            # If we have a path, use the existing function
            if rgb_path and os.path.exists(rgb_path):
                return extract_2d_features_from_image(rgb_path)

            # Otherwise, work with the numpy array
            # Save temporarily to use existing function
            temp_path = "/tmp/temp_rgb.png"
            # Convert from normalized back to 0-255 if needed
            if rgb_image.max() <= 1.0:
                rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)
            else:
                rgb_image_uint8 = rgb_image.astype(np.uint8)

            cv2.imwrite(temp_path, cv2.cvtColor(rgb_image_uint8, cv2.COLOR_RGB2BGR))
            features = extract_2d_features_from_image(temp_path)

            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

            return features

        except Exception as e:
            logger.warning(f"Failed to extract 2D features: {e}")
            return self._get_default_2d_features()

    def _get_default_2d_features(self) -> Dict[str, float]:
        """Return default 2D features in case of extraction failure."""
        return {
            'projected_area': 0.0,
            'perimeter': 0.0,
            'width': 0.0,
            'height': 0.0,
            'convex_hull_area': 0.0,
            'minor_axis_length': 0.0,
            'major_axis_length': 0.0,
            'convex_hull_perimeter': 0.0,
            'approx_area': 0.0,
            'approx_perimeter': 0.0,
            'area_ratio_rect': 0.0,
            'area_ratio_hull': 0.0,
            'max_convexity_defect': 0.0,
            'sum_convexity_defects': 0.0,
            'equiv_diameter': 0.0
        }


class FeatureExtractor3D:
    """Extract 3D features from depth data."""

    def __init__(self, config: Dict):
        """
        Initialize 3D feature extractor.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def extract(self, depth_data: np.ndarray, depth_path: str = None) -> Dict[str, float]:
        """
        Extract 3D features from depth data.

        Args:
            depth_data: Depth data as numpy array
            depth_path: Optional path to save/load from (for compatibility)

        Returns:
            Dictionary of 3D features
        """
        try:
            # If we have a path, use the existing function
            if depth_path and os.path.exists(depth_path):
                return extract_3d_features_from_depth(depth_path)

            # Otherwise, work with the numpy array
            # Save temporarily to use existing function
            temp_path = "/tmp/temp_depth.npy"
            np.save(temp_path, depth_data)
            features = extract_3d_features_from_depth(temp_path)

            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

            return features

        except Exception as e:
            logger.warning(f"Failed to extract 3D features: {e}")
            return self._get_default_3d_features()

    def _get_default_3d_features(self) -> Dict[str, float]:
        """Return default 3D features in case of extraction failure."""
        return {
            'Feature17_ApproxVolume': 0.0,
            'Feature18_MaxDepth': 0.0,
            'Feature19_MinDepth': 0.0,
            'Feature20_AvgDepth': 0.0,
            'Feature21_DepthRange': 0.0,
            'Feature22_StdDepth': 0.0,
            'Feature23_SumDepth': 0.0,
            'Feature24_MinMinusAvg': 0.0,
            'Feature25_MaxMinusAvg': 0.0
        }


class FeatureExtractorResNet:
    """ResNet feature extractor using pretrained ResNet50."""

    def __init__(self, config: Dict):
        """
        Initialize ResNet feature extractor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config['features'].get('resnet_model', 'resnet50')
        self.feature_dim = 2048  # ResNet50 feature dimension

        # Load pretrained ResNet50
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_resnet_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_resnet_model(self):
        from torchvision import models
        model = models.resnet50(pretrained=True)
        # Remove FC layers to get 2048-dim features
        model = torch.nn.Sequential(*list(model.children())[:-1])
        return model

    def _preprocess_image(self, rgb_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess RGB image for ResNet.

        Args:
            rgb_image: RGB image as numpy array (H, W, C)

        Returns:
            Preprocessed torch.Tensor of shape (1, 3, 224, 224)
        """
        img = Image.fromarray(rgb_image.astype(np.uint8))
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
        return img_tensor

    def extract(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Extract ResNet features from RGB image.

        Args:
            rgb_image: RGB image as numpy array (H, W, C)

        Returns:
            Feature vector of shape (2048,)
        """
        try:
            # Preprocess the image
            img_tensor = self._preprocess_image(rgb_image).to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)
                features = features.view(features.size(0), -1).cpu().numpy()[0]

            return features

        except Exception as e:
            logger.warning(f"Failed to extract ResNet features: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)


class FeatureFusion:
    """Combine 2D, 3D, and ResNet features into a single feature vector."""

    def __init__(self, config: Dict):
        """
        Initialize feature fusion module.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.use_2d = config['features'].get('use_2d', True)
        self.use_3d = config['features'].get('use_3d', True)
        self.use_resnet = config['features'].get('use_resnet', True)

    def fuse(self, features_2d: Dict[str, float],
             features_3d: Dict[str, float],
             features_resnet: np.ndarray) -> np.ndarray:
        """
        Fuse multiple feature types into a single vector.

        Args:
            features_2d: Dictionary of 2D features
            features_3d: Dictionary of 3D features
            features_resnet: ResNet feature vector

        Returns:
            Fused feature vector
        """
        feature_parts = []

        # Add 2D features
        if self.use_2d and features_2d:
            feature_2d_values = list(features_2d.values())
            feature_parts.append(np.array(feature_2d_values, dtype=np.float32))

        # Add 3D features
        if self.use_3d and features_3d:
            feature_3d_values = list(features_3d.values())
            feature_parts.append(np.array(feature_3d_values, dtype=np.float32))

        # Add ResNet features
        if self.use_resnet and features_resnet is not None:
            feature_parts.append(features_resnet.astype(np.float32))

        if not feature_parts:
            raise ValueError("No features enabled for fusion")

        # Concatenate all feature parts
        fused_features = np.concatenate(feature_parts)

        return fused_features

    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get dimensions of each feature type.

        Returns:
            Dictionary mapping feature type to dimension
        """
        dims = {}

        if self.use_2d:
            dims['2d'] = 16  # Based on default 2D features
        if self.use_3d:
            dims['3d'] = 9  # Based on default 3D features
        if self.use_resnet:
            dims['resnet'] = 2048  # ResNet50 features

        dims['total'] = sum(dims.values())
        return dims
