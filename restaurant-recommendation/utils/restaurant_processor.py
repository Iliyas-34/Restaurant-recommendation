#!/usr/bin/env python3
"""
Restaurant Data Processor for Smart Filtering
Loads restaurants.json and implements intelligent filtering based on mood, time, and occasion
"""

import json
import os
import logging
from typing import List, Dict, Any
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RestaurantDataProcessor:
    def __init__(self, json_file_path: str = "data/restaurants.json"):
        self.json_file_path = json_file_path
        self.restaurants = []
        self.loaded = False
        
        # Smart filtering mappings
        self.mood_mappings = {
            'happy': {
                'cuisines': ['dessert', 'fast food', 'north indian', 'chinese', 'continental', 'bbq', 'american', 'italian', 'mediterranean'],
                'restaurant_types': ['casual dining', 'fine dining', 'microbrewery', 'bar & restaurant'],
                'keywords': ['celebration', 'party', 'fun', 'vibrant', 'energetic']
            },
            'sad': {
                'cuisines': ['north indian', 'south indian', 'chinese', 'comfort', 'cafe', 'traditional'],
                'restaurant_types': ['casual dining', 'traditional', 'cafe'],
                'keywords': ['comfort', 'homely', 'warm', 'cozy', 'healing']
            },
            'angry': {
                'cuisines': ['spicy', 'chinese', 'mexican', 'thai', 'street food', 'asian', 'indian'],
                'restaurant_types': ['casual dining', 'street food', 'quick bites'],
                'keywords': ['spicy', 'hot', 'bold', 'intense', 'fiery']
            },
            'relaxed': {
                'cuisines': ['continental', 'italian', 'mediterranean', 'cafe', 'european', 'fine dining'],
                'restaurant_types': ['fine dining', 'cafe', 'casual dining'],
                'keywords': ['calm', 'peaceful', 'serene', 'quiet', 'gentle']
            },
            'excited': {
                'cuisines': ['japanese', 'korean', 'thai', 'mexican', 'fusion', 'asian', 'exotic'],
                'restaurant_types': ['fine dining', 'casual dining', 'bar & restaurant'],
                'keywords': ['adventurous', 'bold', 'new', 'unique', 'exciting']
            },
            'bored': {
                'cuisines': ['multi-cuisine', 'fusion', 'continental', 'asian', 'mixed', 'international'],
                'restaurant_types': ['casual dining', 'fine dining', 'multi-cuisine'],
                'keywords': ['variety', 'diverse', 'different', 'interesting', 'creative']
            }
        }
        
        self.time_mappings = {
            'morning': {
                'cuisines': ['south indian', 'north indian', 'continental', 'cafe', 'breakfast', 'traditional'],
                'restaurant_types': ['cafe', 'traditional', 'casual dining'],
                'keywords': ['breakfast', 'morning', 'fresh', 'light', 'energizing']
            },
            'afternoon': {
                'cuisines': ['north indian', 'chinese', 'continental', 'multi-cuisine', 'lunch', 'business'],
                'restaurant_types': ['casual dining', 'business', 'multi-cuisine'],
                'keywords': ['lunch', 'business', 'quick', 'efficient', 'productive']
            },
            'evening': {
                'cuisines': ['north indian', 'continental', 'italian', 'fine dining', 'european', 'romantic'],
                'restaurant_types': ['fine dining', 'casual dining', 'romantic'],
                'keywords': ['dinner', 'romantic', 'intimate', 'elegant', 'sophisticated']
            },
            'night': {
                'cuisines': ['north indian', 'chinese', 'street food', 'late night', 'snacks', 'bar food'],
                'restaurant_types': ['bar & restaurant', 'casual dining', 'late night'],
                'keywords': ['late night', 'snacks', 'comfort', 'casual', 'relaxed']
            }
        }
        
        self.occasion_mappings = {
            'birthday': {
                'cuisines': ['desserts', 'multi-cuisine', 'continental', 'north indian', 'celebration', 'sweet'],
                'restaurant_types': ['fine dining', 'casual dining', 'celebration'],
                'keywords': ['celebration', 'special', 'party', 'festive', 'sweet', 'dessert', 'cake']
            },
            'date': {
                'cuisines': ['continental', 'italian', 'mediterranean', 'fine dining', 'romantic', 'european'],
                'restaurant_types': ['fine dining', 'romantic', 'intimate'],
                'keywords': ['romantic', 'intimate', 'quiet', 'elegant', 'fine dining', 'wine', 'candlelight']
            },
            'party': {
                'cuisines': ['multi-cuisine', 'north indian', 'chinese', 'continental', 'group', 'finger food'],
                'restaurant_types': ['casual dining', 'bar & restaurant', 'group'],
                'keywords': ['group', 'fun', 'vibrant', 'loud', 'energetic', 'celebration', 'festive']
            },
            'family': {
                'cuisines': ['north indian', 'south indian', 'multi-cuisine', 'continental', 'traditional', 'kid-friendly'],
                'restaurant_types': ['casual dining', 'family', 'traditional'],
                'keywords': ['family', 'traditional', 'kid-friendly', 'spacious', 'comfortable']
            },
            'meeting': {
                'cuisines': ['continental', 'multi-cuisine', 'cafe', 'business', 'light'],
                'restaurant_types': ['cafe', 'business', 'casual dining'],
                'keywords': ['quiet', 'professional', 'business', 'formal', 'conference']
            },
            'anniversary': {
                'cuisines': ['continental', 'italian', 'mediterranean', 'fine dining', 'special', 'romantic'],
                'restaurant_types': ['fine dining', 'romantic', 'special'],
                'keywords': ['romantic', 'special', 'elegant', 'fine dining', 'wine', 'intimate']
            }
        }
    
    def load_restaurants(self) -> bool:
        """Load restaurants from JSON file"""
        try:
            if not os.path.exists(self.json_file_path):
                logger.error(f"Restaurants file not found: {self.json_file_path}")
                return False
            
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.restaurants = json.load(f)
            
            logger.info(f"Loaded {len(self.restaurants)} restaurants from {self.json_file_path}")
            self.loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading restaurants: {e}")
            return False
    
    def normalize_restaurant_data(self, restaurant: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize restaurant data for consistent processing"""
        try:
            # Clean and normalize fields
            normalized = {
                'name': str(restaurant.get('Restaurant Name', '')).strip(),
                'cuisines': str(restaurant.get('Cuisines', '')).strip(),
                'location': str(restaurant.get('Location', '')).strip(),
                'city': str(restaurant.get('City', '')).strip(),
                'address': str(restaurant.get('Address', '')).strip(),
                'rating': float(restaurant.get('Aggregate rating', 0)),
                'cost': int(restaurant.get('Average Cost for two', 0)),
                'votes': int(restaurant.get('Votes', 0)),
                'restaurant_type': str(restaurant.get('Restaurant Type', '')).strip(),
                'dish_liked': str(restaurant.get('Dish Liked', '')).strip(),
                'online_order': bool(restaurant.get('Online Order', False)),
                'book_table': bool(restaurant.get('Book Table', False)),
                'price_category': str(restaurant.get('Price Category', '')).strip(),
                'rating_category': str(restaurant.get('Rating Category', '')).strip()
            }
            
            # Clean cuisines - split by comma and clean each
            cuisines_list = [c.strip().lower() for c in normalized['cuisines'].split(',') if c.strip()]
            normalized['cuisines_list'] = cuisines_list
            normalized['cuisines_lower'] = normalized['cuisines'].lower()
            
            # Clean restaurant type
            normalized['restaurant_type_lower'] = normalized['restaurant_type'].lower()
            
            # Clean dish liked
            normalized['dish_liked_lower'] = normalized['dish_liked'].lower()
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing restaurant data: {e}")
            return {}
    
    def calculate_match_score(self, restaurant: Dict[str, Any], mood: str, time: str, occasion: str) -> float:
        """Calculate match score based on mood, time, and occasion"""
        try:
            score = 0.0
            cuisines_lower = restaurant.get('cuisines_lower', '')
            restaurant_type_lower = restaurant.get('restaurant_type_lower', '')
            dish_liked_lower = restaurant.get('dish_liked_lower', '')
            
            # Mood-based scoring (40% weight)
            mood_mapping = self.mood_mappings.get(mood, {})
            mood_score = 0.0
            
            # Check cuisine matches
            for cuisine in mood_mapping.get('cuisines', []):
                if cuisine in cuisines_lower:
                    mood_score += 0.1
            
            # Check restaurant type matches
            for rest_type in mood_mapping.get('restaurant_types', []):
                if rest_type in restaurant_type_lower:
                    mood_score += 0.1
            
            # Check keyword matches in dish liked
            for keyword in mood_mapping.get('keywords', []):
                if keyword in dish_liked_lower:
                    mood_score += 0.05
            
            score += min(mood_score, 0.4)  # Cap at 0.4
            
            # Time-based scoring (30% weight)
            time_mapping = self.time_mappings.get(time, {})
            time_score = 0.0
            
            for cuisine in time_mapping.get('cuisines', []):
                if cuisine in cuisines_lower:
                    time_score += 0.1
            
            for rest_type in time_mapping.get('restaurant_types', []):
                if rest_type in restaurant_type_lower:
                    time_score += 0.1
            
            for keyword in time_mapping.get('keywords', []):
                if keyword in dish_liked_lower:
                    time_score += 0.05
            
            score += min(time_score, 0.3)  # Cap at 0.3
            
            # Occasion-based scoring (30% weight)
            occasion_mapping = self.occasion_mappings.get(occasion, {})
            occasion_score = 0.0
            
            for cuisine in occasion_mapping.get('cuisines', []):
                if cuisine in cuisines_lower:
                    occasion_score += 0.1
            
            for rest_type in occasion_mapping.get('restaurant_types', []):
                if rest_type in restaurant_type_lower:
                    occasion_score += 0.1
            
            for keyword in occasion_mapping.get('keywords', []):
                if keyword in dish_liked_lower:
                    occasion_score += 0.05
            
            score += min(occasion_score, 0.3)  # Cap at 0.3
            
            # Bonus for high ratings
            if restaurant.get('rating', 0) >= 4.5:
                score += 0.1
            
            # Bonus for good value (cost vs rating)
            if restaurant.get('rating', 0) >= 4.0 and restaurant.get('cost', 0) <= 1000:
                score += 0.05
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating match score: {e}")
            return 0.0
    
    def get_smart_recommendations(self, mood: str, time: str, occasion: str, limit: int = 12) -> List[Dict[str, Any]]:
        """Get smart recommendations based on mood, time, and occasion"""
        if not self.loaded:
            if not self.load_restaurants():
                return []
        
        try:
            # Normalize all restaurants
            normalized_restaurants = []
            for restaurant in self.restaurants:
                normalized = self.normalize_restaurant_data(restaurant)
                if normalized and normalized.get('name'):
                    normalized_restaurants.append(normalized)
            
            logger.info(f"Processing {len(normalized_restaurants)} restaurants for filters: {mood}, {time}, {occasion}")
            
            # Calculate match scores for all restaurants
            scored_restaurants = []
            for restaurant in normalized_restaurants:
                match_score = self.calculate_match_score(restaurant, mood, time, occasion)
                restaurant['match_score'] = match_score
                scored_restaurants.append(restaurant)
            
            # Sort by match score (descending) and then by rating (descending)
            scored_restaurants.sort(key=lambda x: (x['match_score'], x['rating']), reverse=True)
            
            # Filter out restaurants with very low scores (less than 0.1)
            filtered_restaurants = [r for r in scored_restaurants if r['match_score'] >= 0.1]
            
            # If we don't have enough high-scoring restaurants, include some top-rated ones
            if len(filtered_restaurants) < limit:
                top_rated = [r for r in scored_restaurants if r['rating'] >= 4.0 and r['match_score'] < 0.1]
                filtered_restaurants.extend(top_rated[:limit - len(filtered_restaurants)])
            
            # Return top recommendations
            recommendations = filtered_restaurants[:limit]
            
            logger.info(f"Returning {len(recommendations)} recommendations")
            for i, rec in enumerate(recommendations[:3]):
                logger.info(f"  {i+1}. {rec['name']} - Score: {rec['match_score']:.3f}, Rating: {rec['rating']}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting smart recommendations: {e}")
            return []
    
    def get_fallback_recommendations(self, limit: int = 12) -> List[Dict[str, Any]]:
        """Get fallback recommendations (top-rated restaurants)"""
        if not self.loaded:
            if not self.load_restaurants():
                return []
        
        try:
            # Normalize all restaurants
            normalized_restaurants = []
            for restaurant in self.restaurants:
                normalized = self.normalize_restaurant_data(restaurant)
                if normalized and normalized.get('name'):
                    normalized_restaurants.append(normalized)
            
            # Sort by rating (descending) and votes (descending)
            normalized_restaurants.sort(key=lambda x: (x['rating'], x['votes']), reverse=True)
            
            return normalized_restaurants[:limit]
            
        except Exception as e:
            logger.error(f"Error getting fallback recommendations: {e}")
            return []

# Global instance
restaurant_processor = RestaurantDataProcessor()

def initialize_restaurant_processor():
    """Initialize the restaurant processor"""
    return restaurant_processor.load_restaurants()

def get_smart_recommendations(mood: str, time: str, occasion: str, limit: int = 12):
    """Get smart recommendations using the global processor"""
    return restaurant_processor.get_smart_recommendations(mood, time, occasion, limit)

def get_fallback_recommendations(limit: int = 12):
    """Get fallback recommendations using the global processor"""
    return restaurant_processor.get_fallback_recommendations(limit)
