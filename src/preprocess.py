"""Data preprocessing module for delivery delay prediction."""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
from src.utils import load_data, setup_logging

logger = logging.getLogger(__name__)


def validate_features(df: pd.DataFrame, feature_columns: list) -> None:
    """Validate that all required feature columns exist in the dataframe.
    
    Args:
        df: Input DataFrame
        feature_columns: List of required feature column names
        
    Raises:
        ValueError: If any required column is missing
    """
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def prepare_features(df: pd.DataFrame, feature_columns: list = FEATURE_COLUMNS) -> pd.DataFrame:
    """Extract and prepare features from the dataset.
    
    This function ensures we only use features available BEFORE dispatch.
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature column names to extract
        
    Returns:
        DataFrame with selected features
        
    Raises:
        ValueError: If required columns are missing
    """
    logger.info("Preparing features...")
    
    # Validate that all required columns exist
    validate_features(df, feature_columns)
    
    # Extract only the approved features
    features_df = df[feature_columns].copy()
    
    # Log basic statistics
    logger.info(f"Selected {len(feature_columns)} features")
    logger.info(f"Feature columns: {feature_columns}")
    logger.info(f"Shape: {features_df.shape}")
    
    # Check for missing values
    missing_counts = features_df.isnull().sum()
    if missing_counts.any():
        logger.warning(f"Missing values detected:\n{missing_counts[missing_counts > 0]}")
    
    return features_df


def prepare_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.Series:
    """Extract target variable from the dataset.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        
    Returns:
        Series with target values
        
    Raises:
        ValueError: If target column is missing
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    target = df[target_column].copy()
    
    # Convert to binary if needed (ensure 0/1 encoding)
    if target.dtype != int:
        target = target.astype(int)
    
    # Validate target values
    unique_values = target.unique()
    if not set(unique_values).issubset({0, 1}):
        logger.warning(f"Unexpected target values: {unique_values}. Converting to binary.")
        target = (target > 0).astype(int)
    
    logger.info(f"Target distribution:\n{target.value_counts()}")
    logger.info(f"Target percentage: {target.mean() * 100:.2f}% delayed")
    
    return target


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and testing sets.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified splitting
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data (test_size={test_size}, stratify={stratify})...")
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param,
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Training target distribution:\n{y_train.value_counts()}")
    logger.info(f"Test target distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test


def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load dataset and prepare features and target for modeling.
    
    Args:
        data_path: Path to the CSV data file
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    setup_logging()
    logger.info("Starting data loading and preparation...")
    
    # Load data
    df = load_data(data_path)
    
    # Prepare features and target
    X = prepare_features(df)
    y = prepare_target(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    logger.info("Data preparation completed successfully")
    
    return X_train, X_test, y_train, y_test
