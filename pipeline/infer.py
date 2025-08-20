
#!/usr/bin/env python3
"""
Inference script for broiler weight estimation pipeline.
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path

# Add pipeline to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pipeline import BroilerWeightPipeline
from pipeline.utils import validate_image_pair, setup_logging

logger = setup_logging()


def infer_single_pair(rgb_path: str, depth_path: str, config_path: str, 
                     model_type: str = None) -> float:
    """
    Run inference on a single RGB-Depth pair.
    
    Args:
        rgb_path: Path to RGB image
        depth_path: Path to depth file
        config_path: Path to configuration file
        model_type: Model type to use
        
    Returns:
        Predicted weight in kg
    """
    # Validate inputs
    is_valid, error = validate_image_pair(rgb_path, depth_path)
    if not is_valid:
        raise ValueError(f"Invalid image pair: {error}")
    
    # Initialize pipeline
    pipeline = BroilerWeightPipeline(config_path)
    
    # Run inference
    prediction = pipeline.predict_from_images(rgb_path, depth_path, model_type)
    
    return prediction


def infer_directory(rgb_dir: str, depth_dir: str, config_path: str, 
                   output_path: str = None, model_type: str = None) -> pd.DataFrame:
    """
    Run inference on all RGB-Depth pairs in directories.
    
    Args:
        rgb_dir: Directory containing RGB images
        depth_dir: Directory containing depth files
        config_path: Path to configuration file
        output_path: Optional output CSV path
        model_type: Model type to use
        
    Returns:
        DataFrame with predictions
    """
    # Initialize pipeline
    pipeline = BroilerWeightPipeline(config_path)
    
    # Run inference
    results = pipeline.predict_from_directory(rgb_dir, depth_dir, model_type)
    
    # Save results if output path specified
    if output_path:
        results.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    
    return results


def main():
    """Main function for inference CLI."""
    parser = argparse.ArgumentParser(description="Run broiler weight estimation inference")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rgb-image", help="Path to single RGB image")
    group.add_argument("--rgb-dir", help="Directory containing RGB images")
    
    parser.add_argument("--depth-file", help="Path to single depth file (required with --rgb-image)")
    parser.add_argument("--depth-dir", help="Directory containing depth files (required with --rgb-dir)")
    
    # Configuration and model options
    parser.add_argument("--config", default="pipeline/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model-type", choices=["lgbm", "xgb"],
                       help="Model type to use (default: from config)")
    
    # Output options
    parser.add_argument("--output", help="Output CSV file for batch inference")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.rgb_image:
            # Single image inference
            if not args.depth_file:
                parser.error("--depth-file is required when using --rgb-image")
            
            prediction = infer_single_pair(
                args.rgb_image, args.depth_file, args.config, args.model_type
            )
            
            print(f"RGB Image: {args.rgb_image}")
            print(f"Depth File: {args.depth_file}")
            print(f"Predicted Weight: {prediction:.3f} kg")
            
        else:
            # Directory inference
            if not args.depth_dir:
                parser.error("--depth-dir is required when using --rgb-dir")
            
            results = infer_directory(
                args.rgb_dir, args.depth_dir, args.config, args.output, args.model_type
            )
            
            print(f"Processed {len(results)} image pairs")
            print(f"Mean predicted weight: {results['predicted_weight_kg'].mean():.3f} kg")
            print(f"Weight range: {results['predicted_weight_kg'].min():.3f} - "
                  f"{results['predicted_weight_kg'].max():.3f} kg")
            
            if not args.output:
                print("\nFirst 10 predictions:")
                print(results.head(10).to_string(index=False))
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
