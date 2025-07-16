"""
Compute features for repeat purchase prediction using Featuretools.
"""

import pandas as pd
import numpy as np
import featuretools as ft
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple


def load_historical_data() -> pd.DataFrame:
    """Load the historical events data."""
    data_path = Path(__file__).parent.parent / "data" / "historical_events.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Historical data not found at {data_path}. "
            "Please run data/generate_data.py first."
        )
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def create_entityset(events_df: pd.DataFrame) -> ft.EntitySet:
    """
    Create a Featuretools EntitySet from the events data.
    """
    es = ft.EntitySet(id="ecommerce")
    
    # Add events entity
    es = es.add_dataframe(
        dataframe_name="events",
        dataframe=events_df,
        index="event_id",
        make_index=True,
        time_index="timestamp"
    )
    
    # Create users entity
    users_df = events_df[['user_id']].drop_duplicates().reset_index(drop=True)
    es = es.add_dataframe(
        dataframe_name="users",
        dataframe=users_df,
        index="user_id"
    )
    
    # Create products entity
    products_df = events_df[['product_id']].drop_duplicates().reset_index(drop=True)
    es = es.add_dataframe(
        dataframe_name="products",
        dataframe=products_df,
        index="product_id"
    )
    
    # Add relationships
    es = es.add_relationship("users", "user_id", "events", "user_id")
    es = es.add_relationship("products", "product_id", "events", "product_id")
    
    return es


def generate_features(es: ft.EntitySet, cutoff_times: pd.DataFrame) -> pd.DataFrame:
    """
    Generate features using Featuretools Deep Feature Synthesis (DFS).
    """
    
    # Define aggregation primitives for feature generation
    agg_primitives = [
        "sum",
        "mean", 
        "count",
        "max",
        "min"
    ]
    
    # Define transformation primitives
    trans_primitives = [
        "day",
        "month", 
        "weekday"
    ]
    
    print("Running Deep Feature Synthesis...")
    
    # Sample cutoff_times for faster processing (first 1000 examples)
    if len(cutoff_times) > 1000:
        print(f"Sampling 1000 examples from {len(cutoff_times)} for faster processing...")
        cutoff_times_sample = cutoff_times.sample(n=1000, random_state=42)
    else:
        cutoff_times_sample = cutoff_times
    
    # Generate features
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="users",
        cutoff_time=cutoff_times_sample,
        agg_primitives=agg_primitives,
        trans_primitives=trans_primitives,
        max_depth=1,  # Reduced from 2 to 1 for faster processing
        verbose=True
    )
    
    return feature_matrix, feature_defs


def create_cutoff_times(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cutoff times for feature generation.
    For each user-product pair, we'll use the first purchase as the cutoff time.
    """
    purchases = events_df[events_df['event_type'] == 'purchase'].copy()
    
    # Get first purchase for each user-product pair
    first_purchases = purchases.groupby(['user_id', 'product_id'])['timestamp'].min().reset_index()
    first_purchases.columns = ['user_id', 'product_id', 'cutoff_time']
    
    # Only keep user_id and cutoff_time for featuretools
    cutoff_times = first_purchases[['user_id', 'cutoff_time']].drop_duplicates()
    
    # Rename columns to match Featuretools requirements
    cutoff_times = cutoff_times.rename(columns={
        'user_id': 'instance_id',
        'cutoff_time': 'time'
    })
    
    return cutoff_times, first_purchases


def create_labels(events_df: pd.DataFrame, first_purchases: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary labels for repeat purchase prediction.
    Label = 1 if user repurchases the same product within 30 days, else 0.
    """
    labels = []
    
    for _, row in first_purchases.iterrows():
        user_id = row['user_id']
        product_id = row['product_id']
        first_purchase_time = row['cutoff_time']
        
        # Look for repeat purchases within 30 days
        repeat_window_end = first_purchase_time + timedelta(days=30)
        
        repeat_purchases = events_df[
            (events_df['user_id'] == user_id) &
            (events_df['product_id'] == product_id) &
            (events_df['event_type'] == 'purchase') &
            (events_df['timestamp'] > first_purchase_time) &
            (events_df['timestamp'] <= repeat_window_end)
        ]
        
        # Label = 1 if there's at least one repeat purchase, else 0
        will_repurchase = 1 if len(repeat_purchases) > 0 else 0
        
        labels.append({
            'user_id': user_id,
            'product_id': product_id,
            'cutoff_time': first_purchase_time,
            'will_repurchase': will_repurchase
        })
    
    return pd.DataFrame(labels)


def add_manual_features(feature_matrix: pd.DataFrame, events_df: pd.DataFrame, 
                       cutoff_times_with_products: pd.DataFrame) -> pd.DataFrame:
    """
    Add manually engineered features that might be harder to generate with DFS.
    """
    manual_features = []
    
    for _, row in cutoff_times_with_products.iterrows():
        user_id = row['user_id']
        product_id = row['product_id']
        cutoff_time = row['cutoff_time']
        
        # Get user's history up to cutoff time
        user_history = events_df[
            (events_df['user_id'] == user_id) &
            (events_df['timestamp'] <= cutoff_time)
        ]
        
        user_purchases = user_history[user_history['event_type'] == 'purchase']
        user_views = user_history[user_history['event_type'] == 'view']
        
        # Product-specific features
        product_history = user_history[user_history['product_id'] == product_id]
        product_purchases = product_history[product_history['event_type'] == 'purchase']
        product_views = product_history[product_history['event_type'] == 'view']
        
        # Calculate features
        total_spend = user_purchases['amount'].sum() if len(user_purchases) > 0 else 0
        avg_transaction_amount = user_purchases['amount'].mean() if len(user_purchases) > 0 else 0
        purchase_count = len(user_purchases)
        view_count = len(user_views)
        
        # Days since last purchase
        if len(user_purchases) > 0:
            last_purchase_date = user_purchases['timestamp'].max()
            days_since_last_purchase = (cutoff_time - last_purchase_date).days
        else:
            days_since_last_purchase = -1  # No previous purchases
        
        # Product-specific features
        product_view_count = len(product_views)
        product_purchase_count = len(product_purchases)
        
        # Repeat purchase count (for this product before cutoff)
        repeat_purchase_count = max(0, product_purchase_count - 1)  # Exclude the current purchase
        
        # View-to-purchase ratio
        view_to_purchase_ratio = view_count / max(1, purchase_count)
        
        manual_features.append({
            'user_id': user_id,
            'product_id': product_id,
            'total_spend': total_spend,
            'avg_transaction_amount': avg_transaction_amount,
            'days_since_last_purchase': days_since_last_purchase,
            'repeat_purchase_count': repeat_purchase_count,
            'purchase_count': purchase_count,
            'view_count': view_count,
            'product_view_count': product_view_count,
            'product_purchase_count': product_purchase_count,
            'view_to_purchase_ratio': view_to_purchase_ratio
        })
    
    manual_df = pd.DataFrame(manual_features)
    manual_df.set_index('user_id', inplace=True)
    
    # Merge with existing feature matrix
    combined_features = feature_matrix.join(manual_df, how='left', rsuffix='_manual')
    
    return combined_features


def main():
    """Main feature engineering pipeline."""
    print("Starting feature engineering...")
    
    # Load data
    events_df = load_historical_data()
    print(f"Loaded {len(events_df)} events")
    
    # Create cutoff times and labels
    cutoff_times, first_purchases = create_cutoff_times(events_df)
    labels_df = create_labels(events_df, first_purchases)
    
    print(f"Created {len(cutoff_times)} training examples")
    print(f"Positive class ratio: {labels_df['will_repurchase'].mean():.3f}")
    
    # Create EntitySet
    es = create_entityset(events_df)
    print(f"Created EntitySet with {len(es.dataframes)} entities")
    
    # Generate features using DFS
    feature_matrix, feature_defs = generate_features(es, cutoff_times)
    print(f"Generated {len(feature_matrix.columns)} features using DFS")
    
    # Add manual features
    feature_matrix = add_manual_features(feature_matrix, events_df, first_purchases)
    print(f"Total features after manual engineering: {len(feature_matrix.columns)}")
    
    # Merge with labels
    training_data = feature_matrix.reset_index().merge(
        labels_df[['user_id', 'product_id', 'will_repurchase']], 
        on='user_id', 
        how='left'
    )
    
    # Handle missing values
    training_data = training_data.fillna(0)
    
    # Save features
    output_path = Path(__file__).parent / "training_features.csv"
    training_data.to_csv(output_path, index=False)
    print(f"\nTraining features saved to: {output_path}")
    
    # Show feature summary
    print(f"\nFeature Summary:")
    print(f"- Total samples: {len(training_data)}")
    print(f"- Total features: {len(training_data.columns) - 3}")  # Exclude user_id, product_id, will_repurchase
    print(f"- Positive class ratio: {training_data['will_repurchase'].mean():.3f}")
    
    print("\nSample features:")
    print(training_data.head())
    
    print("\nFeature columns:")
    feature_cols = [col for col in training_data.columns if col not in ['user_id', 'product_id', 'will_repurchase']]
    print(feature_cols)


if __name__ == "__main__":
    main()
