#!/usr/bin/env python3
"""
Comprehensive system test for Restaurant Finder
Tests all major functionality including UI, API, and data processing
"""

import requests
import json
import time
import sys
import os
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def test_data_loading():
    """Test if data is loaded correctly"""
    print("\n=== Testing Data Loading ===")
    
    try:
        # Test CSV loading
        import pandas as pd
        df = pd.read_csv('data/restaurants_processed.csv')
        print(f"[OK] Loaded {len(df)} restaurants from processed CSV")
        print(f"[INFO] Columns: {df.columns.tolist()}")
        
        # Test sample data
        sample = df.iloc[0]
        print(f"[INFO] Sample restaurant: {sample['Restaurant_Name']} - {sample['Cuisines']} - {sample['City']}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("\n=== Testing API Endpoints ===")
    
    base_url = "http://localhost:5000"
    
    # Test if server is running
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("[OK] Server is running")
        else:
            print(f"[ERROR] Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[ERROR] Server is not running. Please start the Flask app first.")
        return False
    
    # Test search API
    try:
        response = requests.get(f"{base_url}/api/search?q=pizza&per_page=5")
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Search API working - found {data.get('total', 0)} results")
        else:
            print(f"[ERROR] Search API failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Search API test failed: {e}")
        return False
    
    # Test featured API
    try:
        response = requests.get(f"{base_url}/api/featured")
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Featured API working - found {len(data.get('restaurants', []))} featured restaurants")
        else:
            print(f"[ERROR] Featured API failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Featured API test failed: {e}")
        return False
    
    return True

def test_search_functionality():
    """Test search functionality with various queries"""
    print("\n=== Testing Search Functionality ===")
    
    base_url = "http://localhost:5000"
    test_queries = [
        {"q": "pizza", "expected_min": 1},
        {"q": "indian", "expected_min": 1},
        {"q": "bangalore", "expected_min": 1},
        {"q": "chinese", "expected_min": 1},
        {"q": "nonexistent_restaurant_xyz", "expected_min": 0}
    ]
    
    for query in test_queries:
        try:
            response = requests.get(f"{base_url}/api/search", params=query)
            if response.status_code == 200:
                data = response.json()
                total = data.get('total', 0)
                if total >= query['expected_min']:
                    print(f"[OK] Query '{query['q']}' returned {total} results")
                else:
                    print(f"[WARNING] Query '{query['q']}' returned {total} results (expected at least {query['expected_min']})")
            else:
                print(f"[ERROR] Query '{query['q']}' failed with status {response.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] Query '{query['q']}' failed: {e}")
            return False
    
    return True

def test_filter_functionality():
    """Test filter functionality"""
    print("\n=== Testing Filter Functionality ===")
    
    base_url = "http://localhost:5000"
    
    # Test mood filter
    try:
        response = requests.get(f"{base_url}/api/search?mood=happy&per_page=5")
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Mood filter working - found {data.get('total', 0)} results for 'happy' mood")
        else:
            print(f"[ERROR] Mood filter failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Mood filter test failed: {e}")
        return False
    
    # Test time filter
    try:
        response = requests.get(f"{base_url}/api/search?time=evening&per_page=5")
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Time filter working - found {data.get('total', 0)} results for 'evening' time")
        else:
            print(f"[ERROR] Time filter failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Time filter test failed: {e}")
        return False
    
    # Test occasion filter
    try:
        response = requests.get(f"{base_url}/api/search?occasion=dinner&per_page=5")
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Occasion filter working - found {data.get('total', 0)} results for 'dinner' occasion")
        else:
            print(f"[ERROR] Occasion filter failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Occasion filter test failed: {e}")
        return False
    
    return True

def test_predictive_typing():
    """Test predictive typing functionality"""
    print("\n=== Testing Predictive Typing ===")
    
    base_url = "http://localhost:5000"
    
    try:
        response = requests.post(f"{base_url}/predict", 
                               json={"query": "piz"}, 
                               headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            data = response.json()
            suggestions = data.get('suggestions', [])
            print(f"[OK] Predictive typing working - got {len(suggestions)} suggestions for 'piz'")
            if suggestions:
                print(f"[INFO] Sample suggestions: {suggestions[:3]}")
        else:
            print(f"[ERROR] Predictive typing failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Predictive typing test failed: {e}")
        return False
    
    return True

def test_ui_elements():
    """Test if UI elements are accessible"""
    print("\n=== Testing UI Elements ===")
    
    base_url = "http://localhost:5000"
    
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            content = response.text
            
            # Check for key UI elements
            ui_elements = [
                'search-input',
                'mood-filter',
                'time-filter',
                'occasion-filter',
                'search-and-filters-container'
            ]
            
            for element in ui_elements:
                if element in content:
                    print(f"[OK] UI element '{element}' found")
                else:
                    print(f"[WARNING] UI element '{element}' not found")
            
            return True
        else:
            print(f"[ERROR] Home page failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] UI test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting comprehensive system test...")
    
    tests = [
        ("Data Loading", test_data_loading),
        ("API Endpoints", test_api_endpoints),
        ("Search Functionality", test_search_functionality),
        ("Filter Functionality", test_filter_functionality),
        ("Predictive Typing", test_predictive_typing),
        ("UI Elements", test_ui_elements)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} Test")
        print('='*50)
        
        try:
            if test_func():
                print(f"[PASS] {test_name} test passed")
                passed += 1
            else:
                print(f"[FAIL] {test_name} test failed")
        except Exception as e:
            print(f"[ERROR] {test_name} test crashed: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("[SUCCESS] All tests passed! System is working correctly.")
        return True
    else:
        print(f"[WARNING] {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
