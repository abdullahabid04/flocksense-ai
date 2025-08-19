#!/usr/bin/env python3
"""
Script to combine broiler features (2048 features) with chicken features (25 features + weight)
Creates a combined dataset with 2073 features and weight as target variable.
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_chicken_id(chicken_id_str):
    """Extract numeric chicken ID from string format"""
    match = re.search(r'chicken-(\d+)', str(chicken_id_str))
    return int(match.group(1)) if match else None

def load_and_prepare_datasets():
    """Load and prepare both datasets for merging"""
    logger.info("Loading broiler features dataset...")
    try:
        # Load broiler features (2048 features)
        broiler_df = pd.read_csv('broiler_features (1).csv')
        logger.info(f"Broiler features shape: {broiler_df.shape}")
        
        # Extract numeric chicken ID
        broiler_df['chicken_id_numeric'] = broiler_df['Chicken_ID'].apply(extract_chicken_id)
        
        # Remove rows where chicken ID couldn't be extracted
        broiler_df = broiler_df.dropna(subset=['chicken_id_numeric'])
        broiler_df['chicken_id_numeric'] = broiler_df['chicken_id_numeric'].astype(int)
        
        logger.info(f"Broiler features after processing: {broiler_df.shape}")
        logger.info(f"Unique chicken IDs in broiler data: {broiler_df['chicken_id_numeric'].nunique()}")
        
    except Exception as e:
        logger.error(f"Error loading broiler features: {e}")
        return None, None
    
    logger.info("Loading chicken features dataset...")
    try:
        # Load chicken features (25 features + identifiers + weight)
        chicken_df = pd.read_csv('chicken_features_with_identifiers.csv')
        logger.info(f"Chicken features shape: {chicken_df.shape}")
        logger.info(f"Unique chicken IDs in chicken data: {chicken_df['chicken_id'].nunique()}")
        
    except Exception as e:
        logger.error(f"Error loading chicken features: {e}")
        return None, None
    
    return broiler_df, chicken_df

def combine_datasets(broiler_df, chicken_df):
    """Combine the two datasets based on chicken_id"""
    logger.info("Combining datasets...")
    
    # Select feature columns from broiler dataset (exclude identifiers)
    broiler_feature_cols = [col for col in broiler_df.columns if col.startswith('Feature_')]
    logger.info(f"Number of broiler features: {len(broiler_feature_cols)}")
    
    # Select feature columns from chicken dataset (exclude identifiers, keep weight)
    chicken_feature_cols = [col for col in chicken_df.columns 
                          if col not in ['record_id', 'chicken_id', 'session_id', 'batch_id', 'age_days', 'instance_number']]
    logger.info(f"Chicken feature columns: {chicken_feature_cols}")
    
    # Prepare broiler features for merge
    broiler_features = broiler_df[['chicken_id_numeric', 'Image_ID'] + broiler_feature_cols].copy()
    broiler_features.rename(columns={'chicken_id_numeric': 'chicken_id'}, inplace=True)
    
    # Prepare chicken features for merge
    chicken_features = chicken_df[['chicken_id'] + chicken_feature_cols].copy()
    
    # Perform the merge
    logger.info("Performing merge on chicken_id...")
    combined_df = pd.merge(broiler_features, chicken_features, on='chicken_id', how='inner')
    
    logger.info(f"Combined dataset shape: {combined_df.shape}")
    logger.info(f"Number of matched records: {len(combined_df)}")
    
    # Verify we have the expected number of features
    feature_cols = [col for col in combined_df.columns if col not in ['chicken_id', 'Image_ID', 'weight_kg']]
    logger.info(f"Total feature columns: {len(feature_cols)}")
    
    return combined_df

def save_combined_dataset(combined_df, output_file='combined_features_2073.csv'):
    """Save the combined dataset"""
    logger.info(f"Saving combined dataset to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    logger.info("Dataset saved successfully!")
    
    # Print summary statistics
    logger.info("\n=== DATASET SUMMARY ===")
    logger.info(f"Total samples: {len(combined_df)}")
    logger.info(f"Total features: {combined_df.shape[1] - 3}")  # Exclude chicken_id, Image_ID, weight_kg
    logger.info(f"Target variable (weight_kg) - Mean: {combined_df['weight_kg'].mean():.3f}, Std: {combined_df['weight_kg'].std():.3f}")
    logger.info(f"Target variable range: [{combined_df['weight_kg'].min():.3f}, {combined_df['weight_kg'].max():.3f}]")
    logger.info(f"Missing values: {combined_df.isnull().sum().sum()}")
    
    return output_file

def main():
    """Main execution function"""
    logger.info("Starting dataset combination process...")
    
    # Load datasets
    broiler_df, chicken_df = load_and_prepare_datasets()
    if broiler_df is None or chicken_df is None:
        logger.error("Failed to load datasets")
        return
    
    # Combine datasets
    combined_df = combine_datasets(broiler_df, chicken_df)
    if combined_df.empty:
        logger.error("No matching records found between datasets")
        return
    
    # Save combined dataset
    output_file = save_combined_dataset(combined_df)
    
    logger.info(f"Dataset combination completed successfully!")
    logger.info(f"Output file: {output_file}")
    
    return output_file

if __name__ == "__main__":
    main()