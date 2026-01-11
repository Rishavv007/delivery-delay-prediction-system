"""Model definition and pipeline construction for delivery delay prediction."""

import logging
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify numerical and categorical features.
    
    Args:
        X: Feature DataFrame
        
    Returns:
        Tuple of (numerical_features, categorical_features)
    """
    numerical_features = []
    categorical_features = []
    
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            # Check if it's actually categorical (low cardinality integers)
            if X[col].dtype == 'int64' and X[col].nunique() < 20:
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        else:
            categorical_features.append(col)
    
    logger.info(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    return numerical_features, categorical_features


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build preprocessing and modeling pipeline.
    
    The pipeline includes:
    - Missing value imputation (mean for numerical, most_frequent for categorical)
    - OneHotEncoding for categorical features
    - Logistic Regression classifier
    
    Args:
        X: Feature DataFrame to determine feature types
        
    Returns:
        Scikit-learn Pipeline
    """
    logger.info("Building preprocessing and modeling pipeline...")
    
    # Identify feature types
    numerical_features, categorical_features = identify_feature_types(X)
    
    # Create preprocessing transformers
    preprocessors = []
    
    # Numerical feature preprocessing
    if numerical_features:
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
        ])
        preprocessors.append(('num', numerical_transformer, numerical_features))
    
    # Categorical feature preprocessing
    if categorical_features:
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
        ])
        preprocessors.append(('cat', categorical_transformer, categorical_features))
    
    # Combine preprocessors using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=preprocessors,
        remainder='passthrough'
    )
    
    # Build full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    logger.info("Pipeline created successfully")
    logger.info(f"Pipeline steps: {[step[0] for step in pipeline.steps]}")
    
    return pipeline


def create_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Create and train the delivery delay prediction model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained pipeline
    """
    logger.info("Creating model...")
    
    # Build pipeline
    pipeline = build_pipeline(X_train)
    
    # Train model
    logger.info("Training model...")
    pipeline.fit(X_train, y_train)
    logger.info("Model training completed")
    
    return pipeline
