"""
Comprehensive Feature Extraction Script
Extracts 2D and 3D features from all segmented image and depth data pairs
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import feature functions
from feature_functions import extract_geometric_features, preprocess_image
from features_3d import extract_3d_features


def get_chicken_info_from_path(folder_path: str) -> Dict[str, str]:
    """
    Extract chicken information from folder path.
    Example: 20250725_162902_Friday_chicken-180_batch-36_chicken-age-36
    """
    folder_name = os.path.basename(folder_path)
    parts = folder_name.split('_')
    
    try:
        # Extract date and time
        date_str = f"{parts[0]}_{parts[1]}"
        day_of_week = parts[2]
        time_str = parts[3]
        
        # Extract chicken info - handle the combined parts
        chicken_batch_age = '_'.join(parts[4:])  # chicken-180_batch-36_chicken-age-36
        
        # Parse the combined string
        if 'chicken-' in chicken_batch_age and 'batch-' in chicken_batch_age and 'chicken-age-' in chicken_batch_age:
            # Extract chicken ID
            chicken_start = chicken_batch_age.find('chicken-') + 8
            batch_start = chicken_batch_age.find('batch-')
            chicken_id = chicken_batch_age[chicken_start:batch_start-1]
            
            # Extract batch ID
            batch_start += 6
            age_start = chicken_batch_age.find('chicken-age-')
            batch_id = chicken_batch_age[batch_start:age_start-1]
            
            # Extract age
            age_start += 12
            chicken_age = chicken_batch_age[age_start:]
            
        else:
            # Fallback parsing
            chicken_id = "unknown"
            batch_id = "unknown"
            chicken_age = "unknown"
        
        return {
            'date': date_str,
            'day_of_week': day_of_week,
            'time': time_str,
            'chicken_id': chicken_id,
            'batch_id': batch_id,
            'chicken_age': chicken_age,
            'folder_name': folder_name
        }
    except Exception as e:
        print(f"Error parsing folder name '{folder_name}': {e}")
        return {
            'date': 'unknown',
            'day_of_week': 'unknown',
            'time': 'unknown',
            'chicken_id': 'unknown',
            'batch_id': 'unknown',
            'chicken_age': 'unknown',
            'folder_name': folder_name
        }


def extract_instance_info_from_filename(filename: str) -> Dict[str, str]:
    """
    Extract instance information from filename.
    Example: rgb_20250725_162941_594732_instance-0.png
    """
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')
    
    # Extract timestamp and instance
    timestamp = f"{parts[1]}_{parts[2]}_{parts[3]}"
    instance_id = parts[4].split('-')[1]
    
    return {
        'timestamp': timestamp,
        'instance_id': instance_id,
        'filename': filename
    }


def load_depth_data(npy_path: str) -> np.ndarray:
    """Load depth data from .npy file."""
    try:
        depth_data = np.load(npy_path)
        return depth_data
    except Exception as e:
        print(f"Error loading depth data from {npy_path}: {e}")
        return None


def create_mask_from_depth(depth_data: np.ndarray) -> np.ndarray:
    """Create binary mask from depth data where non-zero values are foreground."""
    if depth_data is None:
        return None
    
    # Create binary mask (non-zero values are foreground)
    mask = (depth_data > 0).astype(np.uint8)
    return mask


def extract_2d_features_from_image(png_path: str) -> Dict[str, float]:
    """Extract 2D geometric features from PNG image."""
    try:
        # Preprocess image and get contour
        image, binary, contour = preprocess_image(png_path)
        
        # Extract geometric features
        features_2d = extract_geometric_features(contour, pixel_to_mm=1.0)
        
        return features_2d
        
    except Exception as e:
        print(f"Error extracting 2D features from {png_path}: {e}")
        # Return default values if extraction fails
        return {
            'projected_area': 0.0,
            'perimeter': 0.0,
            'width': 0.0,
            'height': 0.0,
            'convex_hull_area': 0.0,
            'minor_axis_length': 0.0,
            'major_axis_length': 0.0,
            'eccentricity': 0.0,
            'convex_hull_perimeter': 0.0,
            'approx_area': 0.0,
            'approx_perimeter': 0.0,
            'area_ratio_rect': 0.0,
            'area_ratio_hull': 0.0,
                         'max_convexity_defect': 0.0,
             'sum_convexity_defects': 0.0,
             'equiv_diameter': 0.0
        }


def extract_3d_features_from_depth(npy_path: str) -> Dict[str, float]:
    """Extract 3D features from depth data."""
    try:
        # Load depth data
        depth_data = load_depth_data(npy_path)
        if depth_data is None:
            raise ValueError("Could not load depth data")
        
        # Create mask from depth data
        mask = create_mask_from_depth(depth_data)
        if mask is None:
            raise ValueError("Could not create mask from depth data")
        
        # Get segmented depth (only foreground pixels)
        segmented_depth = depth_data[mask > 0]
        
        if segmented_depth.size == 0:
            raise ValueError("No foreground pixels found in depth data")
        
        # Extract 3D features
        features_3d = extract_3d_features(segmented_depth, mask)
        
        return features_3d
        
    except Exception as e:
        print(f"Error extracting 3D features from {npy_path}: {e}")
        # Return default values if extraction fails
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


def find_matching_files(chicken_folder: str) -> List[Tuple[str, str]]:
    """
    Find all PNG and NPY file pairs in a chicken folder.
    Returns list of (png_path, npy_path) tuples.
    """
    png_files = glob.glob(os.path.join(chicken_folder, "*.png"))
    npy_files = glob.glob(os.path.join(chicken_folder, "*.npy"))
    
    # Create dictionaries for easy lookup
    png_dict = {}
    npy_dict = {}
    
    for png_file in png_files:
        base_name = os.path.splitext(os.path.basename(png_file))[0]
        png_dict[base_name] = png_file
    
    for npy_file in npy_files:
        base_name = os.path.splitext(os.path.basename(npy_file))[0]
        npy_dict[base_name] = npy_file
    
    # Find matching pairs
    matching_pairs = []
    for base_name in png_dict:
        if base_name in npy_dict:
            matching_pairs.append((png_dict[base_name], npy_dict[base_name]))
    
    return matching_pairs


def process_chicken_folder(chicken_folder: str) -> List[Dict]:
    """
    Process all file pairs in a chicken folder and extract features.
    Returns list of feature dictionaries.
    """
    chicken_info = get_chicken_info_from_path(chicken_folder)
    matching_pairs = find_matching_files(chicken_folder)
    
    features_list = []
    
    for png_path, npy_path in matching_pairs:
        try:
            # Extract instance info from filename
            png_filename = os.path.basename(png_path)
            instance_info = extract_instance_info_from_filename(png_filename)
            
            # Extract 2D features
            features_2d = extract_2d_features_from_image(png_path)
            
            # Extract 3D features
            features_3d = extract_3d_features_from_depth(npy_path)
            
            # Combine all features
            combined_features = {
                # Metadata (only file paths)
                'png_path': png_path,
                'npy_path': npy_path,
                
                # 2D Features
                **features_2d,
                
                # 3D Features
                **features_3d
            }
            
            features_list.append(combined_features)
            
        except Exception as e:
            print(f"Error processing {png_path}: {e}")
            continue
    
    return features_list


def main():
    """Main function to extract features from all chicken folders."""
    segmented_dir = "segmented"
    
    if not os.path.exists(segmented_dir):
        print(f"Error: {segmented_dir} directory not found!")
        return
    
    # Get all chicken folders
    chicken_folders = [f for f in os.listdir(segmented_dir) 
                      if os.path.isdir(os.path.join(segmented_dir, f))]
    
    print(f"Found {len(chicken_folders)} chicken folders to process")
    
    all_features = []
    
    # Process each chicken folder
    for chicken_folder in tqdm(chicken_folders, desc="Processing chicken folders"):
        folder_path = os.path.join(segmented_dir, chicken_folder)
        
        try:
            features_list = process_chicken_folder(folder_path)
            all_features.extend(features_list)
            
            print(f"Processed {chicken_folder}: {len(features_list)} instances")
            
        except Exception as e:
            print(f"Error processing folder {chicken_folder}: {e}")
            continue
    
    # Create DataFrame and save to CSV
    if all_features:
        df = pd.DataFrame(all_features)
        
        # Save to CSV
        output_csv = "all_features_clean.csv"
        df.to_csv(output_csv, index=False)
        
        print(f"\nFeature extraction completed!")
        print(f"Total instances processed: {len(df)}")
        print(f"Output saved to: {output_csv}")
        print(f"Features extracted: {len(df.columns)}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"Total instances processed: {len(df)}")
        print(f"Features extracted: {len(df.columns)}")
        print(f"2D features: 16")
        print(f"3D features: 9")
        
    else:
        print("No features were extracted!")


if __name__ == "__main__":
    main() 