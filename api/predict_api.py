"""
FastAPI service for real-time repeat purchase predictions.
"""

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import asyncio
import aioredis
import json


app = FastAPI(
    title="E-commerce Repeat Purchase Prediction API",
    description="Real-time ML predictions for repeat purchase likelihood",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    user_id: str
    product_id: str


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    user_id: str
    product_id: str
    repeat_purchase_probability: float
    confidence: str
    prediction_timestamp: str
    model_version: str


class ModelService:
    """Service for loading and managing the ML model."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.model_metadata = None
        self.redis_client = None
        
    async def initialize(self):
        """Initialize the model and Redis connection."""
        await self.load_model()
        await self.connect_redis()
        
    async def load_model(self):
        """Load the trained model and metadata."""
        try:
            model_dir = Path(__file__).parent.parent / "model"
            
            # Load model
            model_path = model_dir / "model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            self.model = joblib.load(model_path)
            print(f"✅ Loaded model from {model_path}")
            
            # Load feature names
            feature_names_path = model_dir / "feature_names.pkl"
            if feature_names_path.exists():
                self.feature_names = joblib.load(feature_names_path)
                print(f"✅ Loaded {len(self.feature_names)} feature names")
            
            # Load metadata
            metadata_path = model_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                print(f"✅ Loaded model metadata: {self.model_metadata['model_type']}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    async def connect_redis(self):
        """Connect to Redis for real-time features."""
        try:
            self.redis_client = await aioredis.from_url(
                "redis://localhost:6379", 
                decode_responses=True
            )
            # Test connection
            await self.redis_client.ping()
            print("✅ Connected to Redis for real-time features")
        except Exception as e:
            print(f"⚠️  Redis connection failed: {e}")
            print("Features will be computed on-demand (slower)")
            self.redis_client = None
    
    async def get_user_features(self, user_id: str, product_id: str) -> Dict:
        """Get real-time features for a user-product pair."""
        
        if self.redis_client:
            # Try to get from Redis first (fast path)
            try:
                feature_key = f"features:{user_id}:{product_id}"
                cached_features = await self.redis_client.get(feature_key)
                
                if cached_features:
                    features = json.loads(cached_features)
                    print(f"✅ Retrieved cached features for {user_id}-{product_id}")
                    return features
                    
            except Exception as e:
                print(f"⚠️  Redis lookup failed: {e}")
        
        # Fallback: compute features on-demand (slow path)
        print(f"🔄 Computing features on-demand for {user_id}-{product_id}")
        return await self.compute_features_on_demand(user_id, product_id)
    
    async def compute_features_on_demand(self, user_id: str, product_id: str) -> Dict:
        """
        Compute features on-demand when not available in cache.
        This is a simplified version for demo purposes.
        """
        # For demo: return default feature values
        # In production, this would query a data store for user history
        
        default_features = {
            'COUNT(events)': 5,
            'MAX(events.amount)': 250.0,
            'MEAN(events.amount)': 150.0,
            'MIN(events.amount)': 50.0,
            'SUM(events.amount)': 750.0,
            'total_spend': 750.0,
            'avg_transaction_amount': 150.0,
            'days_since_last_purchase': 5.0,
            'repeat_purchase_count': 1,
            'purchase_count': 3,
            'view_count': 8,
            'product_view_count': 2,
            'product_purchase_count': 1,
            'view_to_purchase_ratio': 2.67
        }
        
        print(f"⚠️  Using default features for {user_id}-{product_id} (demo mode)")
        return default_features
    
    def predict(self, features: Dict) -> tuple:
        """Make prediction using the loaded model."""
        try:
            # Convert features to the expected format
            feature_vector = []
            
            for feature_name in self.feature_names:
                if feature_name in features:
                    feature_vector.append(features[feature_name])
                else:
                    # Use 0 for missing features
                    feature_vector.append(0.0)
            
            # Make prediction
            feature_array = np.array([feature_vector])
            
            # Get probability for class 1 (repeat purchase)
            probability = self.model.predict_proba(feature_array)[0][1]
            
            # Get confidence level
            confidence = "high" if probability > 0.7 or probability < 0.3 else "medium"
            
            return float(probability), confidence
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Global model service instance
model_service = ModelService()


@app.on_event("startup")
async def startup_event():
    """Initialize the model service on startup."""
    print("🚀 Starting E-commerce Prediction API...")
    await model_service.initialize()
    print("✅ API ready for predictions!")


@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown."""
    if model_service.redis_client:
        await model_service.redis_client.close()
    print("👋 API shutdown complete")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "E-commerce Repeat Purchase Prediction API",
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if not model_service.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": model_service.model_metadata.get("model_type", "Unknown") if model_service.model_metadata else "Unknown",
        "n_features": len(model_service.feature_names) if model_service.feature_names else 0,
        "feature_names": model_service.feature_names[:10] if model_service.feature_names else [],  # First 10
        "training_date": model_service.model_metadata.get("training_date", "Unknown") if model_service.model_metadata else "Unknown",
        "model_params": model_service.model.get_params() if hasattr(model_service.model, 'get_params') else {}
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_repeat_purchase(request: PredictionRequest):
    """
    Predict the probability of repeat purchase for a user-product pair.
    """
    if not model_service.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get real-time features
        features = await model_service.get_user_features(request.user_id, request.product_id)
        
        # Make prediction
        probability, confidence = model_service.predict(features)
        
        # Return response
        return PredictionResponse(
            user_id=request.user_id,
            product_id=request.product_id,
            repeat_purchase_probability=probability,
            confidence=confidence,
            prediction_timestamp=datetime.now().isoformat(),
            model_version=model_service.model_metadata.get("training_date", "unknown") if model_service.model_metadata else "unknown"
        )
        
    except Exception as e:
        print(f"❌ Prediction request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """
    Batch prediction endpoint for multiple user-product pairs.
    """
    if not model_service.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for request in requests:
        try:
            # Get features and predict
            features = await model_service.get_user_features(request.user_id, request.product_id)
            probability, confidence = model_service.predict(features)
            
            results.append(PredictionResponse(
                user_id=request.user_id,
                product_id=request.product_id,
                repeat_purchase_probability=probability,
                confidence=confidence,
                prediction_timestamp=datetime.now().isoformat(),
                model_version=model_service.model_metadata.get("training_date", "unknown") if model_service.model_metadata else "unknown"
            ))
            
        except Exception as e:
            print(f"❌ Batch prediction failed for {request.user_id}-{request.product_id}: {e}")
            # Continue with other predictions
            continue
    
    return {"predictions": results, "total": len(results)}


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting FastAPI server...")
    uvicorn.run(
        "predict_api:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )
