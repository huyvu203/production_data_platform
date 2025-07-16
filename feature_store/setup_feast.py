"""
Initialize and manage the Feast feature store.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from feast import FeatureStore


def init_feature_store():
    """Initialize the Feast feature store."""
    print("Initializing Feast feature store...")
    
    # Change to feature store directory
    feature_store_dir = Path(__file__).parent
    os.chdir(feature_store_dir)
    
    # Initialize feature store
    fs = FeatureStore(repo_path=".")
    
    # Apply feature definitions
    print("Applying feature definitions...")
    fs.apply([])
    
    print("Feature store initialized successfully!")
    return fs


def prepare_feature_data():
    """Prepare training features for Feast materialization."""
    print("Preparing feature data for Feast...")
    
    # Load the training features
    features_path = Path(__file__).parent.parent / "features" / "training_features.csv"
    df = pd.read_csv(features_path)
    
    # Add required timestamp columns for Feast
    now = datetime.now()
    df['feature_timestamp'] = now
    df['created_timestamp'] = now
    
    # Ensure we have the right data types
    df['user_id'] = df['user_id'].astype(str)
    
    # Select only the feature columns we defined in features.py
    feature_columns = [
        'user_id', 'feature_timestamp', 'created_timestamp',
        'view_count', 'purchase_count', 'total_spend', 'avg_transaction_amount',
        'view_to_purchase_ratio', 'days_since_last_purchase', 'distinct_products_viewed',
        'distinct_products_purchased', 'avg_days_between_purchases', 'weekend_purchase_ratio',
        'MEAN(events.amount)', 'MAX(events.amount)', 'MIN(events.amount)', 
        'SUM(events.amount)', 'COUNT(events)'
    ]
    
    # Keep only the columns that exist in our dataframe
    available_columns = [col for col in feature_columns if col in df.columns]
    df_features = df[available_columns].copy()
    
    # Save the prepared features
    output_path = Path(__file__).parent / "feature_data.csv"
    df_features.to_csv(output_path, index=False)
    print(f"Feature data saved to: {output_path}")
    
    return df_features


def materialize_features():
    """Materialize features to the online store."""
    print("Materializing features to online store...")
    
    fs = FeatureStore(repo_path=".")
    
    # Materialize features for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        fs.materialize(start_date, end_date)
        print("Features materialized successfully!")
    except Exception as e:
        print(f"Error materializing features: {e}")
        print("This might be expected for the first run without Redis.")


def test_feature_retrieval():
    """Test retrieving features from the feature store."""
    print("Testing feature retrieval...")
    
    fs = FeatureStore(repo_path=".")
    
    # Get a sample user ID from our data
    features_path = Path(__file__).parent.parent / "features" / "training_features.csv"
    df = pd.read_csv(features_path)
    sample_user = df['user_id'].iloc[0]
    
    try:
        # Retrieve online features
        feature_vector = fs.get_online_features(
            features=[
                "user_behavior_features:view_count",
                "user_behavior_features:purchase_count",
                "user_behavior_features:total_spend",
            ],
            entity_rows=[{"user_id": str(sample_user)}],
        )
        
        result_df = feature_vector.to_df()
        print("Successfully retrieved features:")
        print(result_df.head())
        
    except Exception as e:
        print(f"Error retrieving features: {e}")
        print("This might be expected without Redis running.")


if __name__ == "__main__":
    # Step 1: Prepare feature data
    prepare_feature_data()
    
    # Step 2: Initialize feature store
    init_feature_store()
    
    # Step 3: Test feature retrieval (might fail without Redis)
    test_feature_retrieval()
    
    print("\n" + "="*50)
    print("FEAST SETUP COMPLETE!")
    print("="*50)
    print("Next steps:")
    print("1. Start Redis: docker-compose up -d redis")
    print("2. Materialize features: poetry run python feature_store/setup_feast.py")
    print("3. Start the prediction API server")
