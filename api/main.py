"""FastAPI service for delivery delay prediction."""

import logging

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FEATURE_COLUMNS, MODEL_FILE
from src.utils import load_model, determine_risk_level, setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Delivery Delay Prediction API",
    description="API for predicting delivery delays before shipment dispatch",
    version="1.0.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
logger.info("Loading model...")
try:
    model = load_model(MODEL_FILE)
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error(f"Model file not found: {MODEL_FILE}")
    logger.error("Please train the model first using: python -m src.train")
    model = None


class ShipmentRequest(BaseModel):
    """Request model for shipment prediction."""
    
    days_for_shipment_scheduled: float = Field(
        ...,
        alias="Days for shipment (scheduled)",
        description="Number of days scheduled for shipment",
        ge=0
    )
    shipping_mode: str = Field(
        ...,
        alias="Shipping Mode",
        description="Shipping mode (e.g., Standard Class, First Class)"
    )
    order_region: str = Field(
        ...,
        alias="Order Region",
        description="Order region"
    )
    order_country: str = Field(
        ...,
        alias="Order Country",
        description="Order country"
    )
    order_item_quantity: float = Field(
        ...,
        alias="Order Item Quantity",
        description="Quantity of items in the order",
        ge=1
    )
    sales: float = Field(
        ...,
        alias="Sales",
        description="Sales amount",
        ge=0
    )
    
    model_config = ConfigDict(populate_by_name=True)


class PredictionResponse(BaseModel):
    """Response model for shipment prediction."""
    
    delay_probability: float = Field(..., description="Probability of delay (0-1)")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, or HIGH")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "delay_probability": 0.65,
                "risk_level": "MEDIUM"
            }
        }
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Delivery Delay Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict delivery delay",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_delay(request: ShipmentRequest) -> PredictionResponse:
    """Predict delivery delay probability for a shipment.
    
    Args:
        request: Shipment features
        
    Returns:
        Prediction response with delay probability and risk level
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Convert request to DataFrame with correct column names
        input_data = {
            "Days for shipment (scheduled)": [request.days_for_shipment_scheduled],
            "Shipping Mode": [request.shipping_mode],
            "Order Region": [request.order_region],
            "Order Country": [request.order_country],
            "Order Item Quantity": [request.order_item_quantity],
            "Sales": [request.sales],
        }
        
        df = pd.DataFrame(input_data)
        
        # Ensure columns are in the correct order
        df = df[FEATURE_COLUMNS]
        
        # Make prediction
        logger.info(f"Making prediction for shipment: {input_data}")
        delay_proba = model.predict_proba(df)[0, 1]
        risk_level = determine_risk_level(delay_proba)
        
        logger.info(f"Prediction: probability={delay_proba:.4f}, risk_level={risk_level}")
        
        return PredictionResponse(
            delay_probability=round(delay_proba, 4),
            risk_level=risk_level
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    logger.info(f"API documentation available at http://{API_HOST}:{API_PORT}/docs")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
