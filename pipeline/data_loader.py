
"""
Data loading utilities for RGB and Depth image pairs.
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Generator
from .utils import validate_image_pair, get_matching_depth_path, setup_logging

logger = setup_logging()


class ImagePairDataLoader:
    """
    Data loader for RGB and Depth image pairs used in broiler weight estimation.
    """
    
    def __init__(self, rgb_dir: str, depth_dir: str, config: Dict):
        """
        Initialize the data loader.
        
        Args:
            rgb_dir: Directory containing RGB images
            depth_dir: Directory containing depth (.npy) files
            config: Configuration dictionary
        """
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.config = config
        self.image_pairs = self._discover_image_pairs()
        
    def _discover_image_pairs(self) -> List[Tuple[str, str]]:
        """
        Discover valid RGB-Depth image pairs.
        
        Returns:
            List of (rgb_path, depth_path) tuples
        """
        pairs = []
        
        if not os.path.exists(self.rgb_dir):
            logger.warning(f"RGB directory not found: {self.rgb_dir}")
            return pairs
            
        if not os.path.exists(self.depth_dir):
            logger.warning(f"Depth directory not found: {self.depth_dir}")
            return pairs
        
        rgb_files = [f for f in os.listdir(self.rgb_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for rgb_file in rgb_files:
            rgb_path = os.path.join(self.rgb_dir, rgb_file)
            depth_path = get_matching_depth_path(rgb_path, self.depth_dir)
            
            if depth_path:
                is_valid, error = validate_image_pair(rgb_path, depth_path)
                if is_valid:
                    pairs.append((rgb_path, depth_path))
                else:
                    logger.warning(f"Invalid pair {rgb_file}: {error}")
        
        logger.info(f"Found {len(pairs)} valid RGB-Depth pairs")
        return pairs
    
    def load_rgb_image(self, rgb_path: str) -> np.ndarray:
        """
        Load and preprocess RGB image.
        
        Args:
            rgb_path: Path to RGB image
            
        Returns:
            Preprocessed RGB image as numpy array
        """
        image = cv2.imread(rgb_path)
        if image is None:
            raise ValueError(f"Could not load RGB image: {rgb_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if specified
        if 'resize' in self.config['preprocessing']['rgb']:
            target_size = tuple(self.config['preprocessing']['rgb']['resize'])
            image = cv2.resize(image, target_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        if 'normalize_mean' in self.config['preprocessing']['rgb']:
            mean = np.array(self.config['preprocessing']['rgb']['normalize_mean'])
            std = np.array(self.config['preprocessing']['rgb']['normalize_std'])
            image = (image - mean) / std
        
        return image
    
    def load_depth_data(self, depth_path: str) -> np.ndarray:
        """
        Load and preprocess depth data.
        
        Args:
            depth_path: Path to depth .npy file
            
        Returns:
            Preprocessed depth data as numpy array
        """
        depth_data = np.load(depth_path)
        
        # Handle missing values
        fill_value = self.config['preprocessing']['depth'].get('fill_missing', 0.0)
        depth_data = np.nan_to_num(depth_data, nan=fill_value)
        
        # Scale depth values
        scale_factor = self.config['preprocessing']['depth'].get('scale_factor', 1.0)
        depth_data = depth_data * scale_factor
        
        # Clip to max depth
        max_depth = self.config['preprocessing']['depth'].get('max_depth', np.inf)
        depth_data = np.clip(depth_data, 0, max_depth)
        
        return depth_data
    
    def __len__(self) -> int:
        """Return number of image pairs."""
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """
        Get image pair by index.
        
        Returns:
            Tuple of (rgb_image, depth_data, rgb_path, depth_path)
        """
        rgb_path, depth_path = self.image_pairs[idx]
        rgb_image = self.load_rgb_image(rgb_path)
        depth_data = self.load_depth_data(depth_path)
        return rgb_image, depth_data, rgb_path, depth_path
    
    def batch_generator(self, batch_size: int = 32) -> Generator:
        """
        Generate batches of image pairs.
        
        Args:
            batch_size: Size of each batch
            
        Yields:
            Batches of (rgb_batch, depth_batch, rgb_paths, depth_paths)
        """
        for i in range(0, len(self.image_pairs), batch_size):
            batch_pairs = self.image_pairs[i:i + batch_size]
            
            rgb_batch = []
            depth_batch = []
            rgb_paths = []
            depth_paths = []
            
            for rgb_path, depth_path in batch_pairs:
                try:
                    rgb_image = self.load_rgb_image(rgb_path)
                    depth_data = self.load_depth_data(depth_path)
                    
                    rgb_batch.append(rgb_image)
                    depth_batch.append(depth_data)
                    rgb_paths.append(rgb_path)
                    depth_paths.append(depth_path)
                    
                except Exception as e:
                    logger.warning(f"Failed to load pair {rgb_path}, {depth_path}: {e}")
                    continue
            
            if rgb_batch:  # Only yield if we have valid data
                yield (np.array(rgb_batch), depth_batch, rgb_paths, depth_paths)


def load_training_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the training dataset CSV.
    
    Args:
        csv_path: Path to dataset CSV file
        
    Returns:
        Pandas DataFrame with features and labels
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset with {len(df)} samples and {len(df.columns)} columns")
    
    return df
