"""Utility functions for the delivery delay prediction system."""

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import MODEL_FILE


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_model(model_path: Path = MODEL_FILE) -> Pipeline:
    """Load a trained model from disk.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded scikit-learn pipeline
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logging.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    return model


def save_model(model: Pipeline, model_path: Path = MODEL_FILE) -> None:
    """Save a trained model to disk.
    
    Args:
        model: Trained scikit-learn pipeline
        model_path: Path to save the model file
    """
    # Ensure models directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)
    logging.info("Model saved successfully")


def load_data(data_path: Path, sample_size: int = None) -> pd.DataFrame:
    """Load dataset from CSV file with proper encoding handling.
    
    Args:
        data_path: Path to the CSV file
        sample_size: Optional number of rows to sample (for testing)
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logging.info(f"Loading data from {data_path}")
    
    # Try different encodings to handle encoding issues
    encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(data_path, encoding=encoding, low_memory=False)
            logging.info(f"Successfully loaded data with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError("Failed to load data with any supported encoding")
    
    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        logging.info(f"Sampled {len(df)} rows from dataset")
    
    logging.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def determine_risk_level(probability: float) -> str:
    """Determine risk level based on delay probability.
    
    Args:
        probability: Delay probability (0-1)
        
    Returns:
        Risk level: "LOW", "MEDIUM", or "HIGH"
    """
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"
