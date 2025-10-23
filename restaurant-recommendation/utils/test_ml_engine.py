#!/usr/bin/env python3
"""
Test script for the ML recommendation engine
Tests the trained models and API endpoints
"""

import os
import sys
import json
import requests
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ml_engine_direct():
    """Test the ML engine directly"""
    print("🧪 Testing ML Engine Directly...")
    
    try:
        from ml_recommendation_engine import RestaurantRecommendationEngine
        
        # Initialize engine
        engine = RestaurantRecommendationEngine("zomato.csv")
        
        # Load models
        print("Loading trained models...")
        engine.load_models("restaurant_predictor.joblib")
        
        # Test recommendations
        print("\n📊 Testing Recommendations...")
        
        test_cases = [
            {
                "query": "pizza",
                "mood": "happy",
                "time": "evening",
                "occasion": "dinner"
            },
            {
                "query": "spicy Indian curry",
                "mood": "excited",
                "time": "night",
                "occasion": "party"
            },
            {
                "query": "romantic Italian",
                "mood": "relaxed",
                "time": "evening",
                "occasion": "date"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Query: {test_case['query']}")
            print(f"Mood: {test_case['mood']}, Time: {test_case['time']}, Occasion: {test_case['occasion']}")
            
            recommendations = engine.get_hybrid_recommendations(
                test_case['query'],
                test_case['mood'],
                test_case['time'],
                test_case['occasion'],
                n=3
            )
            
            print(f"Found {len(recommendations)} recommendations:")
            for j, rec in enumerate(recommendations, 1):
                print(f"  {j}. {rec['name']}")
                print(f"     Cuisines: {rec['cuisines']}")
                print(f"     Rating: {rec['rating']}/5")
                print(f"     Match Score: {rec['match_score']:.3f}")
                print(f"     Type: {rec['type']}")
                print()
        
        print("✅ Direct ML Engine Test Passed!")
        return True
        
    except Exception as e:
        print(f"❌ Direct ML Engine Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test the Flask API endpoints"""
    print("\n🌐 Testing API Endpoints...")
    
    base_url = "http://localhost:5000"
    
    # Test ML status endpoint
    try:
        print("Testing /ml/status endpoint...")
        response = requests.get(f"{base_url}/ml/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ ML Status: {status}")
        else:
            print(f"❌ ML Status failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to API: {e}")
        print("Make sure the Flask app is running on localhost:5000")
        return False
    
    # Test recommendation endpoint
    try:
        print("\nTesting /recommend endpoint...")
        test_data = {
            "user_input": "pizza",
            "mood": "happy",
            "time": "evening",
            "occasion": "dinner"
        }
        
        response = requests.post(
            f"{base_url}/recommend",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Recommendations received: {len(result.get('recommendations', []))} results")
            
            if result.get('recommendations'):
                print("Sample recommendations:")
                for i, rec in enumerate(result['recommendations'][:2], 1):
                    print(f"  {i}. {rec['name']} - {rec['cuisines']} (Score: {rec['match_score']})")
        else:
            print(f"❌ Recommendation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Recommendation request failed: {e}")
        return False
    
    print("✅ API Endpoints Test Passed!")
    return True

def main():
    """Main test function"""
    print("🚀 Starting ML Recommendation Engine Tests...")
    print("=" * 50)
    
    # Test direct ML engine
    direct_success = test_ml_engine_direct()
    
    # Test API endpoints (only if Flask app is running)
    api_success = test_api_endpoints()
    
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print(f"Direct ML Engine: {'✅ PASSED' if direct_success else '❌ FAILED'}")
    print(f"API Endpoints: {'✅ PASSED' if api_success else '❌ FAILED'}")
    
    if direct_success and api_success:
        print("\n🎉 All tests passed! The ML recommendation system is working correctly.")
        return True
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
