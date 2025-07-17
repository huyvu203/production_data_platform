"""
Real-time feature engineering service that consumes Kafka events 
and maintains user/product features for predictions.
"""

import json
import asyncio
import redis.asyncio as redis
from kafka import KafkaConsumer
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from threading import Thread
import logging


class RealTimeFeatureEngine:
    """
    Real-time feature engineering service that:
    1. Consumes events from Kafka
    2. Maintains sliding windows of user behavior
    3. Computes features incrementally
    4. Updates Redis with fresh features
    """
    
    def __init__(self, 
                 kafka_servers: str = "localhost:9092",
                 kafka_topic: str = "ecommerce-events",
                 redis_host: str = "localhost",
                 redis_port: int = 6379):
        
        self.kafka_servers = kafka_servers
        self.kafka_topic = kafka_topic
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # In-memory storage for sliding windows
        self.user_events = defaultdict(lambda: deque(maxlen=1000))  # Last 1000 events per user
        self.user_purchases = defaultdict(lambda: deque(maxlen=100))  # Last 100 purchases per user
        self.user_views = defaultdict(lambda: deque(maxlen=500))  # Last 500 views per user
        
        # Redis client for feature storage
        self.redis_client = None
        
        # Kafka consumer
        self.consumer = None
        
        # Control flags
        self.running = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize Redis connection and Kafka consumer."""
        await self.connect_redis()
        self.setup_kafka_consumer()
        
    async def connect_redis(self):
        """Connect to Redis for feature storage."""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
            await self.redis_client.ping()
            self.logger.info("✅ Connected to Redis for feature storage")
        except Exception as e:
            self.logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    def setup_kafka_consumer(self):
        """Setup Kafka consumer for event streaming."""
        try:
            self.consumer = KafkaConsumer(
                self.kafka_topic,
                bootstrap_servers=self.kafka_servers,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                key_deserializer=lambda x: x.decode('utf-8'),
                auto_offset_reset='latest',  # Start from newest messages
                enable_auto_commit=True,
                group_id='feature-engine'
            )
            self.logger.info(f"✅ Kafka consumer setup for topic: {self.kafka_topic}")
        except Exception as e:
            self.logger.error(f"❌ Kafka consumer setup failed: {e}")
            raise
    
    def process_event(self, event: Dict):
        """Process a single event and update user features."""
        try:
            user_id = event['user_id']
            product_id = event['product_id']
            event_type = event['event_type']
            amount = event.get('amount', 0.0)
            timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
            
            # Add to sliding windows
            event_data = {
                'product_id': product_id,
                'event_type': event_type,
                'amount': amount,
                'timestamp': timestamp
            }
            
            self.user_events[user_id].append(event_data)
            
            if event_type == 'purchase':
                self.user_purchases[user_id].append(event_data)
            elif event_type == 'view':
                self.user_views[user_id].append(event_data)
            
            # Compute and cache features synchronously
            self.update_user_features_sync(user_id)
            
            self.logger.info(f"📦 Processed {event_type} event: {user_id}-{product_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing event: {e}")
    
    async def update_user_features(self, user_id: str):
        """Compute and cache features for a user."""
        try:
            user_events = list(self.user_events[user_id])
            
            if not user_events:
                return
            
            # Get unique products for this user
            user_products = set(event['product_id'] for event in user_events)
            
            for product_id in user_products:
                features = self.compute_user_product_features(user_id, product_id, user_events)
                await self.cache_features(user_id, product_id, features)
            
        except Exception as e:
            self.logger.error(f"❌ Error updating features for {user_id}: {e}")
    
    def update_user_features_sync(self, user_id: str):
        """Compute and cache features for a user (synchronous version)."""
        try:
            user_events = list(self.user_events[user_id])
            
            if not user_events:
                return
            
            # Get unique products for this user
            user_products = set(event['product_id'] for event in user_events)
            
            for product_id in user_products:
                features = self.compute_user_product_features(user_id, product_id, user_events)
                self.cache_features_sync(user_id, product_id, features)
            
        except Exception as e:
            self.logger.error(f"❌ Error updating features for {user_id}: {e}")
    
    def compute_user_product_features(self, user_id: str, product_id: str, user_events: List[Dict]) -> Dict:
        """Compute the same features as the training pipeline."""
        
        # Filter events for this user (last 30 days)
        cutoff_time = datetime.now() - timedelta(days=30)
        recent_events = [e for e in user_events if e['timestamp'] >= cutoff_time]
        
        purchases = [e for e in recent_events if e['event_type'] == 'purchase']
        views = [e for e in recent_events if e['event_type'] == 'view']
        
        # Product-specific events
        product_events = [e for e in recent_events if e['product_id'] == product_id]
        product_purchases = [e for e in product_events if e['event_type'] == 'purchase']
        product_views = [e for e in product_events if e['event_type'] == 'view']
        
        # Compute features (matching training pipeline)
        purchase_amounts = [e['amount'] for e in purchases if e['amount'] > 0]
        
        features = {
            # Featuretools-style features
            'COUNT(events)': len(recent_events),
            'MAX(events.amount)': max(purchase_amounts) if purchase_amounts else 0.0,
            'MEAN(events.amount)': np.mean(purchase_amounts) if purchase_amounts else 0.0,
            'MIN(events.amount)': min(purchase_amounts) if purchase_amounts else 0.0,
            'SUM(events.amount)': sum(purchase_amounts),
            
            # Manual features
            'total_spend': sum(purchase_amounts),
            'avg_transaction_amount': np.mean(purchase_amounts) if purchase_amounts else 0.0,
            'purchase_count': len(purchases),
            'view_count': len(views),
            'product_view_count': len(product_views),
            'product_purchase_count': len(product_purchases),
            'repeat_purchase_count': max(0, len(product_purchases) - 1),
            'view_to_purchase_ratio': len(views) / max(1, len(purchases)),
        }
        
        # Days since last purchase
        if purchases:
            last_purchase = max(purchases, key=lambda x: x['timestamp'])
            features['days_since_last_purchase'] = (datetime.now() - last_purchase['timestamp']).days
        else:
            features['days_since_last_purchase'] = -1
        
        return features
    
    async def cache_features(self, user_id: str, product_id: str, features: Dict):
        """Cache computed features in Redis."""
        try:
            feature_key = f"features:{user_id}:{product_id}"
            feature_data = {
                **features,
                'last_updated': datetime.now().isoformat(),
                'user_id': user_id,
                'product_id': product_id
            }
            
            # Cache with 1-hour expiration
            await self.redis_client.setex(
                feature_key, 
                3600,  # 1 hour TTL
                json.dumps(feature_data)
            )
            
            self.logger.debug(f"💾 Cached features for {user_id}-{product_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error caching features: {e}")
    
    def cache_features_sync(self, user_id: str, product_id: str, features: Dict):
        """Cache computed features in Redis (synchronous version)."""
        try:
            import redis as sync_redis
            
            # Create a synchronous Redis client for caching from Kafka thread
            sync_client = sync_redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
            
            feature_key = f"features:{user_id}:{product_id}"
            feature_data = {
                **features,
                'last_updated': datetime.now().isoformat(),
                'user_id': user_id,
                'product_id': product_id
            }
            
            # Cache with 1-hour expiration
            sync_client.setex(
                feature_key, 
                3600,  # 1 hour TTL
                json.dumps(feature_data)
            )
            
            self.logger.debug(f"💾 Cached features for {user_id}-{product_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error caching features: {e}")

    def start_consuming(self):
        """Start consuming events from Kafka."""
        self.running = True
        self.logger.info("🚀 Starting real-time feature engine...")
        
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                event = message.value
                self.process_event(event)
                
        except KeyboardInterrupt:
            self.logger.info("⏹️  Stopping feature engine...")
        except Exception as e:
            self.logger.error(f"❌ Consumer error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the feature engine."""
        self.running = False
        if self.consumer:
            self.consumer.close()
        self.logger.info("👋 Feature engine stopped")
    
    async def get_feature_stats(self) -> Dict:
        """Get statistics about cached features."""
        try:
            # Count cached features
            pattern = "features:*"
            keys = await self.redis_client.keys(pattern)
            
            stats = {
                'total_cached_features': len(keys),
                'active_users': len(self.user_events),
                'total_events_processed': sum(len(events) for events in self.user_events.values()),
                'memory_usage': {
                    'user_events': len(self.user_events),
                    'user_purchases': len(self.user_purchases),
                    'user_views': len(self.user_views)
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Error getting stats: {e}")
            return {}


def run_feature_engine():
    """Run the feature engine as a standalone service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Feature Engine")
    parser.add_argument("--kafka-servers", default="localhost:9092", help="Kafka servers")
    parser.add_argument("--kafka-topic", default="ecommerce-events", help="Kafka topic")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    
    args = parser.parse_args()
    
    # Create and run feature engine
    async def main():
        engine = RealTimeFeatureEngine(
            kafka_servers=args.kafka_servers,
            kafka_topic=args.kafka_topic,
            redis_host=args.redis_host,
            redis_port=args.redis_port
        )
        
        # Initialize Redis connection (async)
        await engine.initialize()
        
        # Start consuming in the main thread (synchronous Kafka consumer)
        try:
            engine.start_consuming()
        except KeyboardInterrupt:
            print("\nShutting down...")
            engine.stop()
            if engine.redis_client:
                await engine.redis_client.close()
    
    # Run the async main function
    asyncio.run(main())


if __name__ == "__main__":
    run_feature_engine()
