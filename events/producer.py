"""
Kafka producer for streaming e-commerce events.
"""

import json
import time
import random
from datetime import datetime, timedelta
from kafka import KafkaProducer
from faker import Faker
from typing import Dict, Any
import argparse


class EcommerceEventProducer:
    """Producer for e-commerce events to Kafka."""
    
    def __init__(self, kafka_servers: str = "localhost:9092", topic: str = "ecommerce-events"):
        """
        Initialize the Kafka producer.
        
        Args:
            kafka_servers: Kafka bootstrap servers
            topic: Kafka topic name
        """
        self.topic = topic
        self.fake = Faker()
        
        # Configure Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: x.encode('utf-8'),
            acks='all',  # Wait for all replicas to acknowledge
            retries=3,
            batch_size=16384,
            linger_ms=10,
        )
        
        print(f"Kafka producer initialized for topic: {topic}")
    
    def generate_event(self, user_pool: list = None, product_pool: list = None) -> Dict[str, Any]:
        """
        Generate a synthetic e-commerce event.
        
        Args:
            user_pool: List of user IDs to sample from
            product_pool: List of product IDs to sample from
            
        Returns:
            Dictionary representing an e-commerce event
        """
        # Default pools if not provided
        if user_pool is None:
            user_pool = [f"user_{i:05d}" for i in range(1000)]
        if product_pool is None:
            product_pool = [f"prod_{i:04d}" for i in range(500)]
        
        user_id = random.choice(user_pool)
        product_id = random.choice(product_pool)
        
        # 70% views, 30% purchases
        event_type = random.choices(['view', 'purchase'], weights=[0.7, 0.3])[0]
        
        # Generate amount based on event type
        if event_type == 'purchase':
            amount = round(random.uniform(10, 500), 2)
        else:
            amount = 0.0
        
        event = {
            'user_id': user_id,
            'product_id': product_id,
            'event_type': event_type,
            'amount': amount,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.fake.uuid4(),
            'ip_address': self.fake.ipv4(),
            'user_agent': self.fake.user_agent(),
        }
        
        return event
    
    def send_event(self, event: Dict[str, Any]) -> None:
        """
        Send an event to Kafka.
        
        Args:
            event: Event dictionary to send
        """
        try:
            # Use user_id as the key for partitioning
            key = event['user_id']
            
            # Send to Kafka
            future = self.producer.send(self.topic, key=key, value=event)
            
            # Optional: wait for confirmation (can be removed for higher throughput)
            record_metadata = future.get(timeout=10)
            
            print(f"Sent event: {event['user_id']} - {event['event_type']} - {event['product_id']}")
            
        except Exception as e:
            print(f"Error sending event: {e}")
    
    def send_batch_events(self, num_events: int, delay_ms: int = 100) -> None:
        """
        Send a batch of events with optional delay between events.
        
        Args:
            num_events: Number of events to send
            delay_ms: Delay in milliseconds between events
        """
        print(f"Sending {num_events} events with {delay_ms}ms delay...")
        
        for i in range(num_events):
            event = self.generate_event()
            self.send_event(event)
            
            if delay_ms > 0 and i < num_events - 1:
                time.sleep(delay_ms / 1000.0)
        
        # Flush any remaining messages
        self.producer.flush()
        print(f"Completed sending {num_events} events")
    
    def send_continuous_events(self, events_per_minute: int = 60) -> None:
        """
        Send events continuously at a specified rate.
        
        Args:
            events_per_minute: Target events per minute
        """
        delay_seconds = 60.0 / events_per_minute
        
        print(f"Starting continuous event stream: {events_per_minute} events/minute")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                event = self.generate_event()
                self.send_event(event)
                time.sleep(delay_seconds)
                
        except KeyboardInterrupt:
            print("\nStopping event producer...")
        finally:
            self.producer.close()
    
    def send_custom_event(self, user_id: str, product_id: str, event_type: str, 
                         amount: float = 0.0) -> None:
        """
        Send a custom event with specified parameters.
        
        Args:
            user_id: User identifier
            product_id: Product identifier  
            event_type: Type of event ('view' or 'purchase')
            amount: Transaction amount (for purchases)
        """
        event = {
            'user_id': user_id,
            'product_id': product_id,
            'event_type': event_type,
            'amount': amount,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.fake.uuid4(),
            'ip_address': self.fake.ipv4(),
            'user_agent': self.fake.user_agent(),
        }
        
        self.send_event(event)
    
    def close(self) -> None:
        """Close the producer connection."""
        self.producer.close()
        print("Kafka producer closed")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="E-commerce Event Producer")
    parser.add_argument(
        "--mode", 
        choices=['batch', 'continuous', 'custom'],
        default='batch',
        help="Sending mode: batch, continuous, or custom"
    )
    parser.add_argument(
        "--events", 
        type=int, 
        default=100,
        help="Number of events to send (batch mode)"
    )
    parser.add_argument(
        "--rate", 
        type=int, 
        default=60,
        help="Events per minute (continuous mode)"
    )
    parser.add_argument(
        "--delay", 
        type=int, 
        default=100,
        help="Delay in milliseconds between events (batch mode)"
    )
    parser.add_argument(
        "--kafka-servers", 
        default="localhost:9092",
        help="Kafka bootstrap servers"
    )
    parser.add_argument(
        "--topic", 
        default="ecommerce-events",
        help="Kafka topic name"
    )
    
    # Custom event arguments
    parser.add_argument("--user-id", help="User ID for custom event")
    parser.add_argument("--product-id", help="Product ID for custom event")
    parser.add_argument("--event-type", choices=['view', 'purchase'], help="Event type for custom event")
    parser.add_argument("--amount", type=float, default=0.0, help="Amount for custom event")
    
    args = parser.parse_args()
    
    # Initialize producer
    producer = EcommerceEventProducer(
        kafka_servers=args.kafka_servers,
        topic=args.topic
    )
    
    try:
        if args.mode == 'batch':
            producer.send_batch_events(args.events, args.delay)
            
        elif args.mode == 'continuous':
            producer.send_continuous_events(args.rate)
            
        elif args.mode == 'custom':
            if not all([args.user_id, args.product_id, args.event_type]):
                print("Custom mode requires --user-id, --product-id, and --event-type")
                return
            producer.send_custom_event(
                args.user_id, args.product_id, args.event_type, args.amount
            )
            
    finally:
        producer.close()


if __name__ == "__main__":
    main()
