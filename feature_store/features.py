"""
Feast feature definitions for the e-commerce MVP platform.
"""

from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field, FileSource, PushSource
from feast.types import Float64, Int64


# Entities
user = Entity(
    name="user_id",
    description="User identifier",
)

product = Entity(
    name="product_id", 
    description="Product identifier",
)

# Feature source - our computed features
user_features_source = FileSource(
    name="user_features_source",
    path="../features/training_features.csv",
    timestamp_field="feature_timestamp",
    created_timestamp_column="created_timestamp",
)

# Push source for real-time features
user_features_push_source = PushSource(
    name="user_features_push_source",
    batch_source=user_features_source,
)

# Feature view for user behavior features
user_behavior_features = FeatureView(
    name="user_behavior_features",
    entities=[user],
    ttl=timedelta(days=30),
    schema=[
        Field(name="view_count", dtype=Int64),
        Field(name="purchase_count", dtype=Int64), 
        Field(name="total_spend", dtype=Float64),
        Field(name="avg_transaction_amount", dtype=Float64),
        Field(name="view_to_purchase_ratio", dtype=Float64),
        Field(name="days_since_last_purchase", dtype=Float64),
        Field(name="distinct_products_viewed", dtype=Int64),
        Field(name="distinct_products_purchased", dtype=Int64),
        Field(name="avg_days_between_purchases", dtype=Float64),
        Field(name="weekend_purchase_ratio", dtype=Float64),
        Field(name="MEAN(events.amount)", dtype=Float64),
        Field(name="MAX(events.amount)", dtype=Float64),
        Field(name="MIN(events.amount)", dtype=Float64),
        Field(name="SUM(events.amount)", dtype=Float64),
        Field(name="COUNT(events)", dtype=Int64),
    ],
    source=user_features_source,
    tags={"team": "ml"},
)
