"""Configuration module for delivery delay prediction system."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "DataCoSupplyChainDataset.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "delivery_delay_model.joblib"

# Feature columns (only pre-dispatch features)
FEATURE_COLUMNS = [
    "Days for shipment (scheduled)",
    "Shipping Mode",
    "Order Region",
    "Order Country",
    "Order Item Quantity",
    "Sales",
]

# Target column
TARGET_COLUMN = "Late_delivery_risk"

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Risk thresholds for API
RISK_THRESHOLDS = {
    "HIGH": 0.7,
    "MEDIUM": 0.4,
    "LOW": 0.0,  # Below MEDIUM threshold
}

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
