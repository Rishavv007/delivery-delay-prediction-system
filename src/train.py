"""Training script for delivery delay prediction model."""

import logging
import sys
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

from src.config import DATA_FILE, MODEL_FILE
from src.model import create_model
from src.preprocess import load_and_prepare_data
from src.utils import save_model, setup_logging

logger = logging.getLogger(__name__)


def train_model() -> None:
    """Train the delivery delay prediction model and save it."""
    setup_logging()
    logger.info("=" * 60)
    logger.info("Starting model training")
    logger.info("=" * 60)
    
    try:
        # Load and prepare data
        logger.info("Step 1: Loading and preparing data...")
        X_train, X_test, y_train, y_test = load_and_prepare_data(DATA_FILE)
        
        # Create and train model
        logger.info("Step 2: Training model...")
        model = create_model(X_train, y_train)
        
        # Evaluate model
        logger.info("Step 3: Evaluating model...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Print classification report
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(y_test, y_pred, target_names=['On-time', 'Delayed']))
        
        # Print confusion matrix
        print("\n" + "=" * 60)
        print("CONFUSION MATRIX")
        print("=" * 60)
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n{cm}")
        print(f"\nTrue Negatives: {cm[0, 0]}")
        print(f"False Positives: {cm[0, 1]}")
        print(f"False Negatives: {cm[1, 0]}")
        print(f"True Positives: {cm[1, 1]}")
        
        # Save model
        logger.info("Step 4: Saving model...")
        save_model(model, MODEL_FILE)
        
        logger.info("\n" + "=" * 60)
        logger.info("Model training completed successfully!")
        logger.info(f"Model saved to: {MODEL_FILE}")
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    train_model()
