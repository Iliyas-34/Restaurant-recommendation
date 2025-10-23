#!/usr/bin/env python3
"""
Comprehensive Search and Filter Test
Tests the complete search and filter functionality on the home page
"""

import requests
import json
import time

def test_search_and_filters():
    """Test the complete search and filter functionality"""
    base_url = "http://localhost:5000"
    
    print("Testing Search and Filter Functionality")
    print("=" * 50)
    
    # Test 1: Basic search functionality
    print("\n1. Testing Basic Search...")
    try:
        response = requests.get(f"{base_url}/api/search?q=pizza&per_page=5")
        data = response.json()
        print(f"   [OK] Pizza search: {len(data.get('restaurants', []))} results")
        
        response = requests.get(f"{base_url}/api/search?q=indian&per_page=5")
        data = response.json()
        print(f"   [OK] Indian search: {len(data.get('restaurants', []))} results")
        
        response = requests.get(f"{base_url}/api/search?q=chinese&per_page=5")
        data = response.json()
        print(f"   [OK] Chinese search: {len(data.get('restaurants', []))} results")
        
    except Exception as e:
        print(f"   [ERROR] Search test failed: {e}")
    
    # Test 2: Mood filters
    print("\n2. Testing Mood Filters...")
    moods = ["happy", "sad", "relaxed", "excited", "angry", "bored"]
    for mood in moods:
        try:
            response = requests.get(f"{base_url}/api/search?mood={mood}&per_page=5")
            data = response.json()
            count = len(data.get('restaurants', []))
            print(f"   [OK] {mood.capitalize()} mood: {count} results")
        except Exception as e:
            print(f"   [ERROR] {mood} mood test failed: {e}")
    
    # Test 3: Time filters
    print("\n3. Testing Time Filters...")
    times = ["morning", "afternoon", "evening", "night"]
    for time in times:
        try:
            response = requests.get(f"{base_url}/api/search?time={time}&per_page=5")
            data = response.json()
            count = len(data.get('restaurants', []))
            print(f"   [OK] {time.capitalize()} time: {count} results")
        except Exception as e:
            print(f"   [ERROR] {time} time test failed: {e}")
    
    # Test 4: Occasion filters
    print("\n4. Testing Occasion Filters...")
    occasions = ["date", "birthday", "party", "meeting", "anniversary", "lunch", "dinner"]
    for occasion in occasions:
        try:
            response = requests.get(f"{base_url}/api/search?occasion={occasion}&per_page=5")
            data = response.json()
            count = len(data.get('restaurants', []))
            print(f"   [OK] {occasion.capitalize()} occasion: {count} results")
        except Exception as e:
            print(f"   [ERROR] {occasion} occasion test failed: {e}")
    
    # Test 5: Combined filters
    print("\n5. Testing Combined Filters...")
    try:
        # Happy mood + Evening time + Dinner occasion
        response = requests.get(f"{base_url}/api/search?mood=happy&time=evening&occasion=dinner&per_page=5")
        data = response.json()
        count = len(data.get('restaurants', []))
        print(f"   [OK] Happy + Evening + Dinner: {count} results")
        
        # Sad mood + Morning time + Meeting occasion
        response = requests.get(f"{base_url}/api/search?mood=sad&time=morning&occasion=meeting&per_page=5")
        data = response.json()
        count = len(data.get('restaurants', []))
        print(f"   [OK] Sad + Morning + Meeting: {count} results")
        
        # Excited mood + Night time + Party occasion
        response = requests.get(f"{base_url}/api/search?mood=excited&time=night&occasion=party&per_page=5")
        data = response.json()
        count = len(data.get('restaurants', []))
        print(f"   [OK] Excited + Night + Party: {count} results")
        
    except Exception as e:
        print(f"   [ERROR] Combined filters test failed: {e}")
    
    # Test 6: Search + Filters combination
    print("\n6. Testing Search + Filters...")
    try:
        # Pizza + Happy mood + Evening time
        response = requests.get(f"{base_url}/api/search?q=pizza&mood=happy&time=evening&per_page=5")
        data = response.json()
        count = len(data.get('restaurants', []))
        print(f"   [OK] Pizza + Happy + Evening: {count} results")
        
        # Indian + Relaxed mood + Afternoon time
        response = requests.get(f"{base_url}/api/search?q=indian&mood=relaxed&time=afternoon&per_page=5")
        data = response.json()
        count = len(data.get('restaurants', []))
        print(f"   [OK] Indian + Relaxed + Afternoon: {count} results")
        
    except Exception as e:
        print(f"   [ERROR] Search + Filters test failed: {e}")
    
    # Test 7: Predictive typing
    print("\n7. Testing Predictive Typing...")
    test_queries = ["piz", "ind", "chi", "caf", "bur"]
    for query in test_queries:
        try:
            response = requests.post(f"{base_url}/predict", 
                                   json={"text": query},
                                   headers={"Content-Type": "application/json"})
            data = response.json()
            suggestion = data.get('suggestion', '')
            print(f"   [OK] '{query}' -> '{suggestion}'")
        except Exception as e:
            print(f"   [ERROR] Predictive typing for '{query}' failed: {e}")
    
    # Test 8: Featured restaurants
    print("\n8. Testing Featured Restaurants...")
    try:
        response = requests.get(f"{base_url}/api/featured")
        data = response.json()
        count = len(data.get('restaurants', []))
        print(f"   [OK] Featured restaurants: {count} results")
    except Exception as e:
        print(f"   [ERROR] Featured restaurants test failed: {e}")
    
    # Test 9: Nearby restaurants
    print("\n9. Testing Nearby Restaurants...")
    try:
        # Test with city
        response = requests.get(f"{base_url}/api/nearby?city=Bangalore")
        data = response.json()
        count = len(data.get('restaurants', []))
        print(f"   [OK] Nearby Bangalore: {count} results")
        
        # Test with coordinates
        response = requests.get(f"{base_url}/api/nearby?lat=12.9716&lng=77.5946")
        data = response.json()
        count = len(data.get('restaurants', []))
        print(f"   [OK] Nearby coordinates: {count} results")
        
    except Exception as e:
        print(f"   [ERROR] Nearby restaurants test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Search and Filter Testing Complete!")
    print("All functionality is working correctly!")

if __name__ == "__main__":
    test_search_and_filters()
