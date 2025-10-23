#!/usr/bin/env python3
"""
Filter Visibility Test
Tests the visibility of filter dropdowns and button text
"""

import requests
import time

def test_filter_visibility():
    """Test that the filter elements are properly visible"""
    base_url = "http://localhost:5000"
    
    print("Testing Filter Visibility")
    print("=" * 40)
    
    try:
        # Test if the home page loads correctly
        response = requests.get(base_url)
        if response.status_code == 200:
            print("[OK] Home page loads successfully")
            
            # Check if the page contains the filter elements
            content = response.text
            
            # Check for filter elements in HTML
            if 'id="mood-filter"' in content:
                print("[OK] Mood filter element found")
            else:
                print("[ERROR] Mood filter element not found")
                
            if 'id="time-filter"' in content:
                print("[OK] Time filter element found")
            else:
                print("[ERROR] Time filter element not found")
                
            if 'id="occasion-filter"' in content:
                print("[OK] Occasion filter element found")
            else:
                print("[ERROR] Occasion filter element not found")
                
            if 'filter-btn-inline' in content:
                print("[OK] Find Perfect Match button found")
            else:
                print("[ERROR] Find Perfect Match button not found")
                
            # Check for CSS classes
            if 'filter-select-inline' in content:
                print("[OK] Filter CSS classes present")
            else:
                print("[ERROR] Filter CSS classes missing")
                
            # Check for proper text content
            if 'Any Mood' in content:
                print("[OK] Filter option text present")
            else:
                print("[ERROR] Filter option text missing")
                
            if 'Find Perfect Match' in content:
                print("[OK] Button text present")
            else:
                print("[ERROR] Button text missing")
                
        else:
            print(f"[ERROR] Home page failed to load: {response.status_code}")
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
    
    print("\n" + "=" * 40)
    print("Filter Visibility Test Complete!")
    print("\nTo verify visual appearance:")
    print("1. Open http://localhost:5000 in your browser")
    print("2. Check that filter dropdowns show text clearly")
    print("3. Verify button text is visible")
    print("4. Test in both light and dark themes")

if __name__ == "__main__":
    test_filter_visibility()
