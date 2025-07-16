"""
FastAPI server for real-time repeat purchase predictions.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

# Feast imports
from feast import FeatureStore


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    user_id: str = Field(..., description="User identifier")
    product_id: str = Field(..., description="Product identifier") 
    amount: float = Field(default=0.0, description="Transaction amount")
    event_type: str = Field(..., description="Event type: 'view' or 'purchase'")
    timestamp: Optional[str] = Field(default=None, description="Event timestamp (ISO format)")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_00123",
                "product_id": "prod_0456",
                "amount": 89.99,
                "event_type": "purchase",
                "timestamp": "2024-07-16T10:30:00"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    prediction: int = Field(..., description="Prediction: 1 (will repurchase) or 0 (won't repurchase)")
    confidence: float = Field(..., description="Prediction confidence score (0-1)")
    user_id: str = Field(..., description="User identifier")
    product_id: str = Field(..., description="Product identifier")
    features_used: Dict[str, float] = Field(..., description="Features used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model_loaded: bool
    feast_connected: bool
    timestamp: str


# Initialize FastAPI app
app = FastAPI(
    title="E-commerce Repeat Purchase Prediction API",
    description="API for predicting whether a user will repurchase a product",
    version="1.0.0"
)

# Global variables for model and feature store
model = None
feature_names = None
feature_store = None


def load_model():
    """Load the trained model and feature names."""
    global model, feature_names
    
    model_dir = Path(__file__).parent.parent / "model"
    
    # Load model
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    
    # Load feature names
    feature_names_path = model_dir / "feature_names.pkl"
    if not feature_names_path.exists():
        raise FileNotFoundError(f"Feature names not found at {feature_names_path}")
    
    feature_names = joblib.load(feature_names_path)
    
    print(f"Model loaded with {len(feature_names)} features")


def initialize_feast():
    """Initialize Feast feature store."""
    global feature_store
    
    try:
        feast_repo_path = Path(__file__).parent.parent / "feast" / "feature_repo"
        feature_store = FeatureStore(repo_path=str(feast_repo_path))
        print("Feast feature store initialized")
    except Exception as e:
        print(f"Warning: Could not initialize Feast feature store: {e}")
        feature_store = None


def get_features_from_feast(user_id: str, product_id: str) -> Dict[str, float]:
    """
    Retrieve features from Feast feature store.
    
    Args:
        user_id: User identifier
        product_id: Product identifier
        
    Returns:
        Dictionary of feature values
    """
    if feature_store is None:
        # Return dummy features if Feast is not available
        return get_dummy_features()
    
    try:
        # Create entity DataFrame for feature retrieval
        entity_df = pd.DataFrame([{
            "user_id": user_id,
            "event_timestamp": datetime.now()
        }])
        
        # Retrieve features
        features = feature_store.get_online_features(
            features=[
                "user_features:total_spend",
                "user_features:avg_transaction_amount", 
                "user_features:days_since_last_purchase",
                "user_features:repeat_purchase_count",
                "user_features:purchase_count",
                "user_features:view_count",
                "user_features:product_view_count",
                "user_features:product_purchase_count",
                "user_features:view_to_purchase_ratio",
            ],
            entity_rows=entity_df.to_dict("records")
        ).to_dict()
        
        # Convert to flat dictionary
        feature_dict = {}
        for feature_name, values in features.items():
            if feature_name != "user_id":
                feature_dict[feature_name] = values[0] if values else 0.0
        
        return feature_dict
        
    except Exception as e:
        print(f"Error retrieving features from Feast: {e}")
        return get_dummy_features()


def get_dummy_features() -> Dict[str, float]:
    """Return dummy features when Feast is not available."""
    return {
        "total_spend": 250.0,
        "avg_transaction_amount": 50.0,
        "days_since_last_purchase": 7,
        "repeat_purchase_count": 2,
        "purchase_count": 5,
        "view_count": 20,
        "product_view_count": 3,
        "product_purchase_count": 1,
        "view_to_purchase_ratio": 4.0,
    }


def prepare_features_for_prediction(
    base_features: Dict[str, float],
    request: PredictionRequest
) -> np.ndarray:
    """
    Prepare feature vector for model prediction.
    
    Args:
        base_features: Base features from Feast
        request: Prediction request data
        
    Returns:
        Feature vector ready for model prediction
    """
    # Start with base features
    features = base_features.copy()
    
    # Add on-demand features
    features["amount_vs_avg_ratio"] = (
        request.amount / max(features.get("avg_transaction_amount", 1), 1)
    )
    features["is_purchase_event"] = 1 if request.event_type == "purchase" else 0
    
    # Ensure all required features are present
    feature_vector = []
    for feature_name in feature_names:
        if feature_name in features:
            feature_vector.append(features[feature_name])
        else:
            # Default value for missing features
            feature_vector.append(0.0)
    
    return np.array(feature_vector).reshape(1, -1)


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    load_model()
    initialize_feast()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        feast_connected=feature_store is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_repeat_purchase(request: PredictionRequest):
    """
    Predict whether a user will repurchase a product.
    
    Args:
        request: Prediction request containing user_id, product_id, etc.
        
    Returns:
        Prediction result with confidence score
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Get features from Feast
        base_features = get_features_from_feast(request.user_id, request.product_id)
        
        # Prepare feature vector
        X = prepare_features_for_prediction(base_features, request)
        
        # Make prediction
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        confidence = float(max(prediction_proba))
        
        # Prepare features used for response
        features_dict = {}
        for i, feature_name in enumerate(feature_names):
            features_dict[feature_name] = float(X[0, i])
        
        return PredictionResponse(
            prediction=int(prediction),
            confidence=confidence,
            user_id=request.user_id,
            product_id=request.product_id,
            features_used=features_dict,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    """
    Predict repeat purchase for multiple user-product pairs.
    
    Args:
        requests: List of prediction requests
        
    Returns:
        List of prediction results
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    for request in requests:
        try:
            prediction_result = await predict_repeat_purchase(request)
            results.append(prediction_result)
        except Exception as e:
            # Include error information for failed predictions
            results.append({
                "user_id": request.user_id,
                "product_id": request.product_id,
                "error": str(e)
            })
    
    return results


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    model_dir = Path(__file__).parent.parent / "model"
    metadata_path = model_dir / "model_metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    else:
        return {
            "model_type": type(model).__name__,
            "n_features": len(feature_names) if feature_names else 0,
            "status": "metadata_not_found"
        }


@app.get("/features/{user_id}")
async def get_user_features(user_id: str, product_id: str = "prod_0001"):
    """
    Get current features for a user.
    
    Args:
        user_id: User identifier
        product_id: Product identifier (optional)
        
    Returns:
        Current feature values for the user
    """
    features = get_features_from_feast(user_id, product_id)
    return {
        "user_id": user_id,
        "product_id": product_id,
        "features": features,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
