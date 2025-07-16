# E-commerce MVP: Repeat Purchase Prediction Platform

A complete end-to-end machine learning platform for predicting whether users will repurchase products. This MVP demonstrates modern MLOps practices with real-time feature serving, streaming data, and RESTful prediction APIs.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Feature Store  │    │ Prediction API  │
│                 │    │                 │    │                 │
│ • Raw Events    │───▶│ • Feast         │───▶│ • FastAPI       │
│ • Kafka Stream  │    │ • Redis Online  │    │ • Real-time ML  │
│ • Historical    │    │ • Feature Eng   │    │ • Batch Predict │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
cd ecommerce_mvp
poetry install
poetry shell
```

### 2. Infrastructure Setup

```bash
# Start Kafka, Redis, and monitoring services
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 3. Data Pipeline

```bash
# Generate historical training data
python data/generate_data.py

# Engineer features using Featuretools
python features/compute_features.py

# Train the Random Forest model
python model/train_model.py
```

### 4. Feature Store Setup

```bash
# Initialize Feast feature repository
cd feast/feature_repo
feast apply

# Load features to online store (if needed)
feast materialize-incremental $(date -d "1 day ago" +%Y-%m-%d) $(date +%Y-%m-%d)
```

### 5. Start Prediction API

```bash
# Start FastAPI server
cd server
python app.py

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### 6. Stream Events & Test

```bash
# Start streaming events to Kafka
python events/producer.py --mode continuous --rate 120

# Test prediction API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_00123",
    "product_id": "prod_0456", 
    "amount": 89.99,
    "event_type": "purchase"
  }'
```

## 📊 Expected Response

```json
{
  "prediction": 1,
  "confidence": 0.84,
  "user_id": "user_00123",
  "product_id": "prod_0456",
  "features_used": {
    "total_spend": 450.75,
    "avg_transaction_amount": 75.12,
    "days_since_last_purchase": 5,
    "repeat_purchase_count": 3
  },
  "timestamp": "2024-07-16T10:30:00.123456"
}
```

## 🗂️ Project Structure

```
ecommerce_mvp/
├── data/
│   ├── generate_data.py           # Synthetic data generation
│   └── historical_events.csv      # Generated training data
├── features/
│   ├── compute_features.py        # Feature engineering with Featuretools
│   └── training_features.csv      # Processed features for training
├── model/
│   ├── train_model.py             # Model training pipeline
│   ├── model.pkl                  # Trained RandomForest model
│   ├── feature_names.pkl          # Feature names for consistency
│   └── model_metadata.json        # Model versioning info
├── feast/
│   └── feature_repo/
│       ├── feature_store.py       # Feast feature definitions
│       └── feature_store.yaml     # Feast configuration
├── events/
│   └── producer.py                # Kafka event streaming
├── server/
│   └── app.py                     # FastAPI prediction service
├── docker-compose.yml             # Infrastructure services
├── pyproject.toml                 # Poetry dependencies
└── README.md                      # This file
```

## 🔧 Key Features

### Data Generation
- **Realistic E-commerce Events**: Views and purchases with proper temporal patterns
- **User Behavior Simulation**: Repeat purchases, seasonal trends, product affinity
- **Configurable Scale**: Easily adjust users, products, and event volumes

### Feature Engineering
- **Automated Feature Discovery**: Featuretools Deep Feature Synthesis (DFS)
- **User-Level Features**: Total spend, average transaction, purchase frequency
- **Product-Level Features**: Product views, purchase history, repeat patterns
- **Temporal Features**: Days since last purchase, time-based aggregations

### Model Training
- **RandomForest Classifier**: Robust, interpretable model for tabular data
- **Class Imbalance Handling**: Balanced class weights and proper evaluation
- **Feature Importance**: Understand which features drive predictions
- **Cross-Validation**: Robust model evaluation with multiple folds

### Feature Store (Feast)
- **Online Features**: Redis-backed real-time feature serving
- **Offline Features**: File-based batch feature storage
- **On-Demand Features**: Real-time feature computation
- **Feature Versioning**: Consistent features across training and serving

### Real-Time Streaming
- **Kafka Integration**: Scalable event streaming platform
- **JSON Payloads**: Structured event format with metadata
- **Partitioning**: Events partitioned by user_id for scalability
- **Monitoring**: Kafdrop UI for Kafka topic monitoring

### Prediction API
- **FastAPI Framework**: Modern, fast API with automatic documentation
- **Real-Time Predictions**: Sub-second response times
- **Batch Predictions**: Process multiple requests efficiently  
- **Feature Transparency**: Return features used for each prediction
- **Health Monitoring**: Endpoint for service health checks

## 📈 Monitoring & Operations

### Service Health
```bash
# Check API health
curl http://localhost:8000/health

# View model information
curl http://localhost:8000/model/info

# Get user features
curl "http://localhost:8000/features/user_00123?product_id=prod_0456"
```

### Kafka Monitoring
- **Kafdrop UI**: http://localhost:9000
- **Topics**: Monitor `ecommerce-events` topic
- **Consumer Lag**: Track processing delays

### Feature Store Status
```bash
cd feast/feature_repo
feast ui  # Start Feast UI for feature monitoring
```

## 🎯 Business Value

### Use Cases
1. **Real-Time Personalization**: Show targeted offers to high-repeat-probability users
2. **Inventory Planning**: Predict demand for repeat purchases
3. **Customer Retention**: Identify users likely to churn (low repeat probability)
4. **Marketing Optimization**: Focus campaigns on users with medium repeat probability

### Key Metrics
- **Precision/Recall**: Balance false positives vs false negatives
- **AUC Score**: Overall model discrimination ability
- **Feature Importance**: Understand drivers of repeat purchases
- **Prediction Latency**: Real-time response requirements

## 🔄 Development Workflow

### Adding New Features
1. Update `features/compute_features.py` with new feature logic
2. Retrain model with `python model/train_model.py`
3. Update Feast feature definitions in `feast/feature_repo/`
4. Deploy updated model and feature store

### Model Retraining
```bash
# Automated retraining pipeline
python data/generate_data.py      # Refresh training data
python features/compute_features.py  # Recompute features
python model/train_model.py       # Retrain with new data
```

### Production Deployment
- **Containerization**: Dockerize API service and workers
- **Load Balancing**: Multiple API instances behind load balancer
- **Model Versioning**: A/B test new models against baseline
- **Feature Store Scaling**: Distributed Redis cluster for high throughput

## 🚦 API Endpoints

### Core Endpoints
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /health` - Service health check
- `GET /model/info` - Model metadata
- `GET /features/{user_id}` - User feature lookup

### Example Usage
```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict", json={
    "user_id": "user_00123",
    "product_id": "prod_0456",
    "amount": 89.99,
    "event_type": "purchase"
})

prediction = response.json()
print(f"Will repurchase: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']:.2f}")
```

## 🔧 Configuration

### Environment Variables
- `KAFKA_SERVERS`: Kafka bootstrap servers (default: localhost:9092)
- `REDIS_HOST`: Redis host for Feast (default: localhost)
- `REDIS_PORT`: Redis port (default: 6379)
- `API_HOST`: FastAPI host (default: 0.0.0.0)
- `API_PORT`: FastAPI port (default: 8000)

### Docker Compose Services
- **Kafka + Zookeeper**: Event streaming platform
- **Redis**: Feature store online storage
- **Kafdrop**: Kafka monitoring UI

## 🎓 Learning Outcomes

This MVP demonstrates:
- **Feature Engineering**: Automated feature discovery with Featuretools
- **Model Training**: End-to-end ML pipeline with evaluation
- **Feature Stores**: Online/offline feature serving with Feast
- **Stream Processing**: Real-time event streaming with Kafka
- **API Development**: Production-ready ML APIs with FastAPI
- **MLOps Practices**: Model versioning, monitoring, and deployment

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Machine Learning! 🚀**
