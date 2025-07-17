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
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_00123",
                "product_id": "prod_0456"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    user_id: str = Field(..., description="User identifier")
    product_id: str = Field(..., description="Product identifier")
    repeat_purchase_probability: float = Field(..., description="Probability of repeat purchase (0-1)")
    confidence: str = Field(..., description="Confidence level: high, medium, low")
    prediction_timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version/training date")
    features_used: Dict[str, float] = Field(..., description="Features used for prediction")


class HealthResponse(BaseModel):
    """Health check response model."""
    service: str
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
        
        # Retrieve features for our actual trained model
        features = feature_store.get_online_features(
            features=[
                "user_behavior_features:COUNT(events)",
                "user_behavior_features:MAX(events.amount)",
                "user_behavior_features:MEAN(events.amount)",
                "user_behavior_features:MIN(events.amount)",
                "user_behavior_features:SUM(events.amount)",
                "user_behavior_features:total_spend",
                "user_behavior_features:avg_transaction_amount", 
                "user_behavior_features:days_since_last_purchase",
                "user_behavior_features:repeat_purchase_count",
                "user_behavior_features:purchase_count",
                "user_behavior_features:view_count",
                "user_behavior_features:product_view_count",
                "user_behavior_features:product_purchase_count",
                "user_behavior_features:view_to_purchase_ratio",
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
        "COUNT(events)": 5.0,
        "MAX(events.amount)": 250.0,
        "MEAN(events.amount)": 150.0,
        "MIN(events.amount)": 50.0,
        "SUM(events.amount)": 750.0,
        "total_spend": 750.0,
        "avg_transaction_amount": 150.0,
        "days_since_last_purchase": 7.0,
        "repeat_purchase_count": 1.0,
        "purchase_count": 3.0,
        "view_count": 8.0,
        "product_view_count": 2.0,
        "product_purchase_count": 1.0,
        "view_to_purchase_ratio": 2.67
    }


def prepare_features_for_prediction(
    base_features: Dict[str, float],
    request: PredictionRequest
) -> np.ndarray:
    """
    Prepare feature vector for model prediction.
    
    Args:
        base_features: Base features from Feast or dummy features
        request: Prediction request data
        
    Returns:
        Feature vector ready for model prediction
    """
    # Use the base features as-is since they match our training features
    features = base_features.copy()
    
    # Ensure all required features are present in the correct order
    feature_vector = []
    for feature_name in feature_names:
        if feature_name in features:
            feature_vector.append(float(features[feature_name]))
        else:
            # Default value for missing features
            print(f"Warning: Missing feature {feature_name}, using default value 0.0")
            feature_vector.append(0.0)
    
    return np.array(feature_vector).reshape(1, -1)


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    load_model()
    initialize_feast()


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        service="E-commerce Repeat Purchase Prediction API",
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
        request: Prediction request containing user_id, product_id
        
    Returns:
        Prediction result with probability and confidence
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Get features from Feast or fallback to dummy features
        base_features = get_features_from_feast(request.user_id, request.product_id)
        
        # Prepare feature vector
        X = prepare_features_for_prediction(base_features, request)
        
        # Make prediction - get probability for repeat purchase (class 1)
        prediction_proba = model.predict_proba(X)[0]
        repeat_purchase_probability = float(prediction_proba[1])
        
        # Determine confidence level
        confidence = "high" if repeat_purchase_probability > 0.7 or repeat_purchase_probability < 0.3 else "medium"
        if 0.4 <= repeat_purchase_probability <= 0.6:
            confidence = "low"
        
        # Prepare features used for response
        features_dict = {}
        for i, feature_name in enumerate(feature_names):
            features_dict[feature_name] = float(X[0, i])
        
        # Load model metadata for version
        model_dir = Path(__file__).parent.parent / "model"
        metadata_path = model_dir / "model_metadata.json"
        model_version = "unknown"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                model_version = metadata.get("training_date", "unknown")
        
        return PredictionResponse(
            user_id=request.user_id,
            product_id=request.product_id,
            repeat_purchase_probability=repeat_purchase_probability,
            confidence=confidence,
            prediction_timestamp=datetime.now().isoformat(),
            model_version=model_version,
            features_used=features_dict
        )
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
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
            print(f"❌ Batch prediction failed for {request.user_id}-{request.product_id}: {e}")
            # Continue with other predictions
            continue
    
    return {"predictions": results, "total": len(results), "timestamp": datetime.now().isoformat()}


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
