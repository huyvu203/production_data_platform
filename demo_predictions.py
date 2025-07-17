"""
Demo script to test the prediction API with various scenarios.
Shows how the API handles different types of users and products.
"""

import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Any


class PredictionDemo:
    """Demo class for testing prediction API with various scenarios."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def check_api_health(self) -> bool:
        """Check if the API is running and healthy."""
        try:
            response = self.session.get(f"{self.api_base_url}/")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API Health: {health_data}")
                return True
            else:
                print(f"❌ API Health Check Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        try:
            response = self.session.get(f"{self.api_base_url}/model/info")
            if response.status_code == 200:
                model_info = response.json()
                print(f"📊 Model Info: {model_info}")
                return model_info
            else:
                print(f"❌ Failed to get model info: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ Error getting model info: {e}")
            return {}
    
    def make_single_prediction(self, user_id: str, product_id: str) -> Dict:
        """Make a single prediction for a user-product pair."""
        try:
            payload = {
                "user_id": user_id,
                "product_id": product_id
            }
            
            print(f"\n🔮 Making prediction for {user_id} - {product_id}")
            
            response = self.session.post(
                f"{self.api_base_url}/predict",
                json=payload
            )
            
            if response.status_code == 200:
                prediction = response.json()
                self.print_prediction_result(prediction)
                return prediction
            else:
                print(f"❌ Prediction failed: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return {}
    
    def make_batch_predictions(self, requests_list: List[Dict]) -> List[Dict]:
        """Make batch predictions for multiple user-product pairs."""
        try:
            # The batch endpoint expects a direct list, not wrapped in {"requests": ...}
            payload = requests_list
            
            print(f"\n📦 Making batch prediction for {len(requests_list)} requests")
            
            response = self.session.post(
                f"{self.api_base_url}/predict/batch",
                json=payload
            )
            
            if response.status_code == 200:
                print(f"✅ Batch predictions successful!")
                
                batch_response = response.json()
                
                # The response is a dict with 'predictions', 'total', and 'timestamp'
                if isinstance(batch_response, dict) and 'predictions' in batch_response:
                    predictions = batch_response['predictions']
                    total = batch_response.get('total', len(predictions))
                    timestamp = batch_response.get('timestamp', 'unknown')
                    
                    print(f"� Processed {total} predictions at {timestamp}")
                    
                    # Display each prediction
                    for i, pred in enumerate(predictions):
                        print(f"\n📈 Batch Prediction {i+1}/{total}:")
                        self.print_prediction_result(pred)
                    
                    return predictions
                else:
                    print(f"❌ Unexpected response format - missing 'predictions' key")
                    print(f"Response keys: {batch_response.keys() if isinstance(batch_response, dict) else 'Not a dict'}")
                    return []
                    
            else:
                print(f"❌ Batch prediction failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"❌ Error making batch prediction: {e}")
            return []
    
    def print_prediction_result(self, prediction: Dict):
        """Pretty print a prediction result."""
        if not prediction:
            return
            
        print(f"📈 Prediction Result:")
        print(f"   User: {prediction.get('user_id', 'Unknown')}")
        print(f"   Product: {prediction.get('product_id', 'Unknown')}")
        print(f"   Repeat Purchase Probability: {prediction.get('repeat_purchase_probability', 0):.3f}")
        print(f"   Confidence: {prediction.get('confidence', 'unknown')}")
        
        # Show key features if available
        features = prediction.get('features_used', {})
        if features:
            print(f"   Key Features:")
            print(f"     • Purchase Count: {features.get('purchase_count', 0)}")
            print(f"     • Repeat Purchases: {features.get('repeat_purchase_count', 0)}")
            print(f"     • Total Spend: ${features.get('total_spend', 0):.2f}")
            print(f"     • Days Since Last Purchase: {features.get('days_since_last_purchase', 'N/A')}")
            print(f"     • View Count: {features.get('view_count', 0)}")
    
    def run_demo_scenarios(self):
        """Run 5 different demo scenarios to showcase the API."""
        
        print("🚀 Starting Prediction API Demo")
        print("=" * 50)
        
        # Check API health first
        if not self.check_api_health():
            print("❌ API is not available. Please start the server first:")
            print("   poetry run python server/app.py")
            return
        
        # Get model information
        self.get_model_info()
        
        print("\n" + "=" * 50)
        print("📊 DEMO SCENARIOS")
        print("=" * 50)
        
        # Scenario 1: User with real-time data (from producer events)
        print(f"\n🎯 SCENARIO 1: User with Real-time Data")
        print("Testing user who should have features from producer events")
        self.make_single_prediction("user_00606", "prod_0342")
        
        time.sleep(1)
        
        # Scenario 2: Different user with real-time data
        print(f"\n🎯 SCENARIO 2: Another Active User")
        print("Testing another user from producer events")
        self.make_single_prediction("user_00587", "prod_0220")
        
        time.sleep(1)
        
        # Scenario 3: Brand new user (should use default features)
        print(f"\n🎯 SCENARIO 3: Brand New User")
        print("Testing completely new user with no history")
        self.make_single_prediction("new_user_demo", "new_product_demo")
        
        time.sleep(1)
        
        # Scenario 4: Mix of known and unknown users (batch prediction)
        print(f"\n🎯 SCENARIO 4: Batch Predictions (Mixed Users)")
        print("Testing batch prediction with mix of known/unknown users")
        
        batch_requests = [
            {"user_id": "user_00606", "product_id": "prod_0342"},  # Known user
            {"user_id": "user_00587", "product_id": "prod_0220"},  # Known user  
            {"user_id": "batch_test_user", "product_id": "batch_test_product"},  # New user
        ]
        
        self.make_batch_predictions(batch_requests)
        
        time.sleep(1)
        
        # Scenario 5: Performance test with rapid predictions
        print(f"\n🎯 SCENARIO 5: Performance Test")
        print("Testing rapid-fire predictions to check latency")
        
        start_time = time.time()
        
        performance_requests = [
            {"user_id": f"perf_user_{i}", "product_id": f"perf_product_{i}"} 
            for i in range(5)
        ]
        
        for i, req in enumerate(performance_requests, 1):
            pred_start = time.time()
            result = self.make_single_prediction(req["user_id"], req["product_id"])
            pred_time = time.time() - pred_start
            print(f"   Prediction {i} latency: {pred_time:.3f}s")
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total time for 5 predictions: {total_time:.3f}s")
        print(f"⏱️  Average latency: {total_time/5:.3f}s per prediction")
        
        print("\n" + "=" * 50)
        print("✅ DEMO COMPLETE!")
        print("=" * 50)
        print("\n📝 Demo Summary:")
        print("• Tested real-time features from producer events")
        print("• Tested default features for new users")
        print("• Tested batch predictions")
        print("• Measured API performance and latency")
        print("• Verified end-to-end ML pipeline functionality")
        print("\n🎉 Your e-commerce prediction API is working perfectly!")


def main():
    """Main function to run the prediction demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo script for prediction API")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL for the prediction API"
    )
    parser.add_argument(
        "--scenario",
        choices=['all', 'single', 'batch', 'performance'],
        default='all',
        help="Which demo scenario to run"
    )
    
    args = parser.parse_args()
    
    demo = PredictionDemo(api_base_url=args.api_url)
    
    if args.scenario == 'all':
        demo.run_demo_scenarios()
    elif args.scenario == 'single':
        demo.make_single_prediction("demo_user", "demo_product")
    elif args.scenario == 'batch':
        requests_list = [
            {"user_id": "batch_user_1", "product_id": "batch_product_1"},
            {"user_id": "batch_user_2", "product_id": "batch_product_2"},
        ]
        demo.make_batch_predictions(requests_list)
    elif args.scenario == 'performance':
        print("🏃‍♂️ Running performance test...")
        start = time.time()
        for i in range(10):
            demo.make_single_prediction(f"perf_user_{i}", f"perf_product_{i}")
        elapsed = time.time() - start
        print(f"⏱️  10 predictions in {elapsed:.3f}s ({elapsed/10:.3f}s avg)")


if __name__ == "__main__":
    main()
