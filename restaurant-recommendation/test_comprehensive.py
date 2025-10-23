#!/usr/bin/env python3
"""
Comprehensive test suite for the restaurant recommendation system
Tests all major functionality including API endpoints, ML models, and UI components
"""

import unittest
import requests
import json
import os
import sys
import time
from unittest.mock import patch, MagicMock

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestRestaurantRecommendationSystem(unittest.TestCase):
    """Comprehensive test suite for the restaurant recommendation system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.base_url = "http://localhost:5000"
        cls.test_restaurant = "Byg Brewski Brewing Company"
        cls.test_city = "Bangalore"
        cls.test_cuisine = "Italian"
        
        # Wait for Flask app to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{cls.base_url}/", timeout=5)
                if response.status_code == 200:
                    print(f"[OK] Flask app is ready after {i+1} attempts")
                    break
            except requests.exceptions.RequestException:
                if i == max_retries - 1:
                    raise Exception("Flask app not ready after 30 attempts")
                time.sleep(1)
    
    def test_homepage_loads(self):
        """Test that the homepage loads correctly"""
        response = requests.get(f"{self.base_url}/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Restaurant Finder", response.text)
        print("[OK] Homepage loads correctly")
    
    def test_restaurants_page_loads(self):
        """Test that the restaurants page loads correctly"""
        response = requests.get(f"{self.base_url}/restaurants")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Find Your Favorite Restaurant", response.text)
        print("[OK] Restaurants page loads correctly")
    
    def test_api_search_basic(self):
        """Test basic search API functionality"""
        response = requests.get(f"{self.base_url}/api/search?q=pizza")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("restaurants", data)
        self.assertGreater(len(data["restaurants"]), 0)
        print("[OK] Basic search API works")
    
    def test_api_search_with_filters(self):
        """Test search API with filters"""
        response = requests.get(f"{self.base_url}/api/search?q=italian&mood=happy&time=evening&occasion=dinner")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("restaurants", data)
        print("[OK] Search API with filters works")
    
    def test_api_restaurant_details(self):
        """Test restaurant details API"""
        response = requests.get(f"{self.base_url}/api/restaurant/{self.test_restaurant}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("name", data)
        self.assertEqual(data["name"], self.test_restaurant)
        print("[OK] Restaurant details API works")
    
    def test_api_nearby_restaurants(self):
        """Test nearby restaurants API"""
        response = requests.get(f"{self.base_url}/api/nearby?city={self.test_city}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("restaurants", data)
        self.assertGreater(len(data["restaurants"]), 0)
        print("[OK] Nearby restaurants API works")
    
    def test_api_featured_restaurants(self):
        """Test featured restaurants API"""
        response = requests.get(f"{self.base_url}/api/featured")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("restaurants", data)
        self.assertGreater(len(data["restaurants"]), 0)
        print("[OK] Featured restaurants API works")
    
    def test_api_recommendations(self):
        """Test ML recommendations API"""
        response = requests.post(f"{self.base_url}/recommend", 
                               json={"user_input": "pizza", "mood": "happy", "time": "evening", "occasion": "dinner"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("recommendations", data)
        print("[OK] ML recommendations API works")
    
    def test_chatbot_status(self):
        """Test chatbot status API"""
        response = requests.get(f"{self.base_url}/chatbot/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("restaurant_count", data)
        print("[OK] Chatbot status API works")
    
    def test_chatbot_query(self):
        """Test chatbot query functionality"""
        response = requests.post(f"{self.base_url}/chatbot", 
                               json={"message": "pizza restaurants"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        print("[OK] Chatbot query API works")
    
    def test_api_filters(self):
        """Test filters API"""
        response = requests.get(f"{self.base_url}/api/filters")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("cities", data)
        self.assertIn("cuisines", data)
        print("[OK] Filters API works")
    
    def test_search_pagination(self):
        """Test search API pagination"""
        response = requests.get(f"{self.base_url}/api/search?q=restaurant&page=1&per_page=5")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("restaurants", data)
        self.assertLessEqual(len(data["restaurants"]), 5)
        print("[OK] Search pagination works")
    
    def test_search_sorting(self):
        """Test search API sorting"""
        response = requests.get(f"{self.base_url}/api/search?q=restaurant&sort_by=rating")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("restaurants", data)
        print("[OK] Search sorting works")
    
    def test_restaurant_details_page(self):
        """Test restaurant details page"""
        response = requests.get(f"{self.base_url}/restaurant/{self.test_restaurant}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Restaurant Details", response.text)
        print("[OK] Restaurant details page works")
    
    def test_wishlist_page(self):
        """Test wishlist page"""
        response = requests.get(f"{self.base_url}/wishlist")
        self.assertEqual(response.status_code, 200)
        print("[OK] Wishlist page works")
    
    def test_admin_dashboard(self):
        """Test admin dashboard"""
        response = requests.get(f"{self.base_url}/admin")
        self.assertEqual(response.status_code, 200)
        print("[OK] Admin dashboard works")
    
    def test_error_handling(self):
        """Test error handling for invalid requests"""
        # Test invalid restaurant
        response = requests.get(f"{self.base_url}/api/restaurant/NonExistentRestaurant")
        self.assertEqual(response.status_code, 404)
        
        # Test invalid search parameters
        response = requests.get(f"{self.base_url}/api/search?invalid_param=test")
        self.assertEqual(response.status_code, 200)  # Should handle gracefully
        print("[OK] Error handling works")
    
    def test_data_consistency(self):
        """Test data consistency across APIs"""
        # Get featured restaurants
        featured_response = requests.get(f"{self.base_url}/api/featured")
        featured_data = featured_response.json()
        
        # Get search results
        search_response = requests.get(f"{self.base_url}/api/search?q=restaurant")
        search_data = search_response.json()
        
        # Both should return restaurant data
        self.assertIn("restaurants", featured_data)
        self.assertIn("restaurants", search_data)
        self.assertGreater(len(featured_data["restaurants"]), 0)
        self.assertGreater(len(search_data["restaurants"]), 0)
        print("[OK] Data consistency across APIs")
    
    def test_performance(self):
        """Test API performance"""
        start_time = time.time()
        response = requests.get(f"{self.base_url}/api/search?q=restaurant")
        end_time = time.time()
        
        self.assertEqual(response.status_code, 200)
        self.assertLess(end_time - start_time, 5.0)  # Should respond within 5 seconds
        print(f"[OK] API performance: {end_time - start_time:.2f}s")
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = requests.options(f"{self.base_url}/api/search")
        self.assertIn("Access-Control-Allow-Origin", response.headers)
        print("[OK] CORS headers present")
    
    def test_content_type(self):
        """Test content type headers"""
        response = requests.get(f"{self.base_url}/api/search?q=test")
        self.assertEqual(response.headers["Content-Type"], "application/json")
        print("[OK] Content type headers correct")

class TestMLModels(unittest.TestCase):
    """Test ML models and recommendation functionality"""
    
    def test_model_loading(self):
        """Test that ML models are loaded correctly"""
        try:
            from app import df, cosine_sim, tfidf
            self.assertIsNotNone(df)
            self.assertIsNotNone(cosine_sim)
            self.assertIsNotNone(tfidf)
            print("[OK] ML models loaded correctly")
        except ImportError:
            self.fail("Failed to import ML models")
    
    def test_recommendation_function(self):
        """Test recommendation function"""
        try:
            from app import recommend_restaurants
            recommendations = recommend_restaurants("pizza", n=5)
            self.assertIsInstance(recommendations, list)
            self.assertLessEqual(len(recommendations), 5)
            print("[OK] Recommendation function works")
        except Exception as e:
            self.fail(f"Recommendation function failed: {e}")

class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and quality"""
    
    def test_restaurant_data_structure(self):
        """Test that restaurant data has expected structure"""
        try:
            from app import restaurants
            self.assertGreater(len(restaurants), 0)
            
            # Check first restaurant has required fields
            first_restaurant = restaurants[0]
            required_fields = ["Restaurant Name", "Cuisines", "Location", "City", "Aggregate rating", "Average Cost for two"]
            for field in required_fields:
                self.assertIn(field, first_restaurant)
            print("[OK] Restaurant data structure is correct")
        except Exception as e:
            self.fail(f"Data structure test failed: {e}")
    
    def test_data_quality(self):
        """Test data quality metrics"""
        try:
            from app import restaurants
            total_restaurants = len(restaurants)
            
            # Check for restaurants with ratings
            restaurants_with_ratings = [r for r in restaurants if r.get("Aggregate rating") and float(r.get("Aggregate rating", 0)) > 0]
            rating_percentage = len(restaurants_with_ratings) / total_restaurants * 100
            
            # Check for restaurants with cuisines
            restaurants_with_cuisines = [r for r in restaurants if r.get("Cuisines") and r.get("Cuisines").strip()]
            cuisine_percentage = len(restaurants_with_cuisines) / total_restaurants * 100
            
            self.assertGreater(rating_percentage, 50)  # At least 50% should have ratings
            self.assertGreater(cuisine_percentage, 80)  # At least 80% should have cuisines
            
            print(f"[OK] Data quality: {rating_percentage:.1f}% have ratings, {cuisine_percentage:.1f}% have cuisines")
        except Exception as e:
            self.fail(f"Data quality test failed: {e}")

def run_tests():
    """Run all tests"""
    print("Starting comprehensive test suite...")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestRestaurantRecommendationSystem,
        TestMLModels,
        TestDataIntegrity
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("=" * 50)
    print(f"Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
