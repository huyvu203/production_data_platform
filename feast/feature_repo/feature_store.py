from feast import Entity, FeatureStore, FeatureView, Field, FileSource, PushSource, RequestSource
from feast.types import Float32, Float64, Int32, Int64, String
from datetime import timedelta
import pandas as pd


# Define entities
user = Entity(name="user_id", value_type=String, description="User identifier")
product = Entity(name="product_id", value_type=String, description="Product identifier")

# Define feature sources
# File source for batch features (training)
user_features_source = FileSource(
    name="user_features_source",
    path="/home/huyvu/Projects/production_data_platform/ecommerce_mvp/features/training_features.csv",
    timestamp_field="event_timestamp",
)

# Push source for real-time features
user_features_push_source = PushSource(
    name="user_features_push_source",
    batch_source=user_features_source,
)

# Define feature views
user_features_fv = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="total_spend", dtype=Float64),
        Field(name="avg_transaction_amount", dtype=Float64),
        Field(name="days_since_last_purchase", dtype=Int64),
        Field(name="repeat_purchase_count", dtype=Int64),
        Field(name="purchase_count", dtype=Int64),
        Field(name="view_count", dtype=Int64),
        Field(name="product_view_count", dtype=Int64),
        Field(name="product_purchase_count", dtype=Int64),
        Field(name="view_to_purchase_ratio", dtype=Float64),
    ],
    online=True,
    source=user_features_push_source,
    tags={"team": "ml_platform"},
)

# On-demand feature view for real-time computation
from feast import on_demand_feature_view

@on_demand_feature_view(
    sources=[
        user_features_fv,
        RequestSource(
            name="request_data",
            schema=[
                Field(name="current_amount", dtype=Float64),
                Field(name="event_type", dtype=String),
            ],
        ),
    ],
    schema=[
        Field(name="amount_vs_avg_ratio", dtype=Float64),
        Field(name="is_purchase_event", dtype=Int32),
    ],
)
def real_time_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute on-demand features for real-time prediction.
    """
    df = pd.DataFrame()
    
    # Ratio of current amount to average transaction amount
    df["amount_vs_avg_ratio"] = (
        inputs["current_amount"] / inputs["avg_transaction_amount"].replace(0, 1)
    )
    
    # Binary indicator for purchase events
    df["is_purchase_event"] = (inputs["event_type"] == "purchase").astype(int)
    
    return df
