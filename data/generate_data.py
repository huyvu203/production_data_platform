"""
Generate synthetic e-commerce event data for training the repeat purchase prediction model.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from pathlib import Path

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)


def generate_users(n_users: int = 1000) -> list:
    """Generate a list of unique user IDs."""
    return [f"user_{i:05d}" for i in range(n_users)]


def generate_products(n_products: int = 500) -> list:
    """Generate a list of unique product IDs."""
    return [f"prod_{i:04d}" for i in range(n_products)]


def generate_events(
    users: list,
    products: list,
    start_date: datetime,
    end_date: datetime,
    n_events: int = 50000,
) -> pd.DataFrame:
    """
    Generate synthetic e-commerce events.
    
    Events include:
    - view: User viewed a product (no amount)
    - purchase: User purchased a product (with amount)
    """
    events = []
    
    # Define product price ranges for realistic amounts
    product_prices = {
        product: round(random.uniform(10, 500), 2) 
        for product in products
    }
    
    for _ in range(n_events):
        user_id = random.choice(users)
        product_id = random.choice(products)
        
        # 70% views, 30% purchases (realistic conversion rate)
        event_type = random.choices(
            ['view', 'purchase'], 
            weights=[0.7, 0.3]
        )[0]
        
        # Generate timestamp
        timestamp = fake.date_time_between(
            start_date=start_date, 
            end_date=end_date
        )
        
        # Set amount based on event type
        if event_type == 'purchase':
            base_price = product_prices[product_id]
            # Add some variation to the price (±20%)
            amount = round(base_price * random.uniform(0.8, 1.2), 2)
        else:
            amount = 0.0
        
        events.append({
            'user_id': user_id,
            'product_id': product_id,
            'event_type': event_type,
            'amount': amount,
            'timestamp': timestamp
        })
    
    # Convert to DataFrame and sort by timestamp
    df = pd.DataFrame(events)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Add some repeat purchases to make the model more interesting
    df = add_repeat_purchases(df, users, products, product_prices)
    
    return df


def add_repeat_purchases(
    df: pd.DataFrame, 
    users: list, 
    products: list, 
    product_prices: dict,
    repeat_probability: float = 0.45
) -> pd.DataFrame:
    """
    Add realistic repeat purchases to make the dataset more interesting.
    Users will repurchase the same product with different patterns:
    - Quick repeat (1-7 days): 30% chance
    - Medium repeat (7-30 days): 25% chance  
    - Long repeat (30-90 days): 20% chance
    """
    repeat_events = []
    
    # Get all purchase events
    purchases = df[df['event_type'] == 'purchase'].copy()
    
    for _, purchase in purchases.iterrows():
        # Multiple chances for repeat purchases with different patterns
        patterns = [
            (0.30, 1, 7),    # Quick repeat: 30% chance, 1-7 days
            (0.25, 7, 30),   # Medium repeat: 25% chance, 7-30 days
            (0.20, 30, 90),  # Long repeat: 20% chance, 30-90 days
        ]
        
        for prob, min_days, max_days in patterns:
            if random.random() < prob:
                # Create repeat purchase within the specified range
                days_later = random.randint(min_days, max_days)
                repeat_timestamp = purchase['timestamp'] + timedelta(days=days_later)
                
                # Make sure repeat purchase is within our date range
                max_date = df['timestamp'].max()
                if repeat_timestamp <= max_date:
                    base_price = product_prices[purchase['product_id']]
                    # Slight price variation for repeat purchases
                    repeat_amount = round(base_price * random.uniform(0.85, 1.15), 2)
                    
                    repeat_events.append({
                        'user_id': purchase['user_id'],
                        'product_id': purchase['product_id'],
                        'event_type': 'purchase',
                        'amount': repeat_amount,
                        'timestamp': repeat_timestamp
                    })
        
        # Additional chance for loyal customers (multiple repeats)
        if random.random() < 0.10:  # 10% of customers are very loyal
            # Create 2-4 additional repeat purchases
            num_repeats = random.randint(2, 4)
            for i in range(num_repeats):
                days_later = random.randint(7, 60) * (i + 1)  # Spaced out repeats
                repeat_timestamp = purchase['timestamp'] + timedelta(days=days_later)
                
                max_date = df['timestamp'].max()
                if repeat_timestamp <= max_date:
                    base_price = product_prices[purchase['product_id']]
                    repeat_amount = round(base_price * random.uniform(0.9, 1.1), 2)
                    
                    repeat_events.append({
                        'user_id': purchase['user_id'],
                        'product_id': purchase['product_id'],
                        'event_type': 'purchase',
                        'amount': repeat_amount,
                        'timestamp': repeat_timestamp
                    })
    
    # Add repeat events to the original dataframe
    if repeat_events:
        repeat_df = pd.DataFrame(repeat_events)
        df = pd.concat([df, repeat_df], ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def main():
    """Generate and save historical e-commerce events."""
    print("Generating synthetic e-commerce data...")
    
    # Configuration - Increased for more repeat purchase opportunities
    n_users = 300  # More users
    n_products = 150  # More products
    n_events = 2000  # More initial events
    
    # Date range: 6 months of historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    # Generate data
    users = generate_users(n_users)
    products = generate_products(n_products)
    
    print(f"Generating {n_events} events for {n_users} users and {n_products} products...")
    events_df = generate_events(users, products, start_date, end_date, n_events)
    
    # Add some data quality info
    purchases_df = events_df[events_df['event_type'] == 'purchase']
    
    print(f"\nGenerated {len(events_df)} total events:")
    print(f"- Views: {len(events_df[events_df['event_type'] == 'view'])}")
    print(f"- Purchases: {len(purchases_df)}")
    print(f"- Date range: {events_df['timestamp'].min()} to {events_df['timestamp'].max()}")
    print(f"- Total revenue: ${purchases_df['amount'].sum():,.2f}")
    
    # Analyze repeat purchases
    user_product_purchases = purchases_df.groupby(['user_id', 'product_id']).size()
    repeat_purchases = user_product_purchases[user_product_purchases > 1]
    total_repeat_count = (user_product_purchases - 1)[user_product_purchases > 1].sum()
    
    print(f"\nRepeat Purchase Analysis:")
    print(f"- Unique user-product combinations with purchases: {len(user_product_purchases)}")
    print(f"- User-product combinations with repeat purchases: {len(repeat_purchases)}")
    print(f"- Total repeat purchase events: {total_repeat_count}")
    print(f"- Repeat purchase rate: {len(repeat_purchases)/len(user_product_purchases)*100:.1f}%")
    print(f"- Users with at least one repeat purchase: {len(repeat_purchases.index.get_level_values('user_id').unique())}")
    print(f"- Total unique users: {len(purchases_df['user_id'].unique())}")
    users_with_repeats = len(repeat_purchases.index.get_level_values('user_id').unique())
    total_purchasing_users = len(purchases_df['user_id'].unique())
    print(f"- % of users who made repeat purchases: {users_with_repeats/total_purchasing_users*100:.1f}%")
    
    # Save to CSV
    output_path = Path(__file__).parent / "historical_events.csv"
    events_df.to_csv(output_path, index=False)
    print(f"\nData saved to: {output_path}")
    
    # Show sample data
    print("\nSample data:")
    print(events_df.head(10))


if __name__ == "__main__":
    main()
