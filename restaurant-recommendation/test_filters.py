#!/usr/bin/env python3
"""
Test script for the smart filters functionality
"""

import requests
import json
import time

def test_recommendation_endpoint():
    """Test the recommendation endpoint"""
    url = "http://localhost:5000/recommend"
    
    test_cases = [
        {
            "user_input": "Italian food",
            "mood": "happy", 
            "time": "evening",
            "occasion": "dinner"
        },
        {
            "user_input": "Chinese food",
            "mood": "excited",
            "time": "afternoon", 
            "occasion": "lunch"
        },
        {
            "user_input": "Indian food",
            "mood": "relaxed",
            "time": "night",
            "occasion": "date"
        }
    ]
    
    print("Testing recommendation endpoint...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {test_case}")
        
        try:
            response = requests.post(url, json=test_case, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"[OK] Success! Status: {response.status_code}")
                print(f"Recommendations: {len(data.get('recommendations', []))}")
                
                if data.get('recommendations'):
                    print("Sample recommendation:")
                    rec = data['recommendations'][0]
                    print(f"  Name: {rec.get('name', 'N/A')}")
                    print(f"  Cuisines: {rec.get('cuisines', 'N/A')}")
                    print(f"  Rating: {rec.get('rating', 'N/A')}")
                    print(f"  Address: {rec.get('address', 'N/A')}")
                    print(f"  Match Score: {rec.get('match_score', 'N/A')}")
            else:
                print(f"[ERROR] Error! Status: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("[ERROR] Connection Error: App not running on localhost:5000")
            break
        except Exception as e:
            print(f"[ERROR] Error: {e}")
    
    print("\n" + "="*50)
    print("Test completed!")

if __name__ == "__main__":
    # Wait a bit for the app to start
    print("Waiting for app to start...")
    time.sleep(3)
    test_recommendation_endpoint()
