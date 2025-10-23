#!/usr/bin/env python3
"""
Dataset-Indexed Q&A Chatbot for Restaurant Finder
Provides accurate answers based on the restaurant dataset without hallucinations
"""

import pandas as pd
import numpy as np
import pickle
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RestaurantChatbot:
    """Dataset-indexed chatbot for restaurant Q&A"""
    
    def __init__(self):
        self.df = None
        self.vectorizer = None
        self.restaurant_vectors = None
        self.cuisine_vectors = None
        self.location_vectors = None
        self.qa_patterns = {}
        self.initialized = False
        
    def load_data(self, csv_path: str) -> None:
        """Load restaurant data"""
        logger.info(f"Loading restaurant data from {csv_path}")
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} restaurants")
        
        # Create comprehensive text features for indexing
        self.df['searchable_text'] = self.df.apply(
            lambda row: f"{row['Restaurant_Name']} {row['Cuisines']} {row['Location']} {row['City']} {row['Address']} {row['Restaurant_Type']} {row['Dish_Liked']}", 
            axis=1
        )
        
        # Clean and prepare text
        self.df['searchable_text'] = self.df['searchable_text'].fillna('').str.lower()
        
    def create_vector_indexes(self) -> None:
        """Create vector indexes for different types of queries"""
        logger.info("Creating vector indexes...")
        
        # Restaurant name and description vectors
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2
        )
        
        self.restaurant_vectors = self.vectorizer.fit_transform(self.df['searchable_text'])
        
        # Cuisine-specific vectors
        cuisine_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.cuisine_vectors = cuisine_vectorizer.fit_transform(self.df['Cuisines'].fillna(''))
        
        # Location-specific vectors
        location_text = self.df['Location'] + ' ' + self.df['City'] + ' ' + self.df['Address']
        location_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.location_vectors = location_vectorizer.fit_transform(location_text.fillna(''))
        
        logger.info("Vector indexes created")
    
    def create_qa_patterns(self) -> None:
        """Create Q&A patterns for common questions"""
        logger.info("Creating Q&A patterns...")
        
        self.qa_patterns = {
            'restaurant_search': [
                r'find.*restaurant.*named\s+(\w+)',
                r'where.*is\s+(\w+.*restaurant)',
                r'about\s+(\w+.*restaurant)',
                r'restaurant\s+(\w+)',
                r'(\w+.*restaurant).*details'
            ],
            'cuisine_search': [
                r'(\w+).*cuisine.*restaurant',
                r'restaurant.*(\w+).*food',
                r'(\w+).*food.*near',
                r'best.*(\w+).*restaurant',
                r'(\w+).*restaurant.*recommend'
            ],
            'location_search': [
                r'restaurant.*in\s+(\w+)',
                r'(\w+).*restaurant.*near',
                r'restaurant.*near\s+(\w+)',
                r'(\w+).*area.*restaurant',
                r'restaurant.*(\w+).*location'
            ],
            'rating_search': [
                r'best.*rated.*restaurant',
                r'highest.*rating.*restaurant',
                r'top.*rated.*restaurant',
                r'restaurant.*high.*rating',
                r'best.*restaurant.*rating'
            ],
            'price_search': [
                r'cheap.*restaurant',
                r'budget.*restaurant',
                r'expensive.*restaurant',
                r'fine.*dining.*restaurant',
                r'restaurant.*price.*range'
            ],
            'feature_search': [
                r'restaurant.*online.*order',
                r'restaurant.*book.*table',
                r'restaurant.*delivery',
                r'restaurant.*takeaway'
            ]
        }
        
        logger.info("Q&A patterns created")
    
    def classify_query(self, query: str) -> Tuple[str, str]:
        """Classify query type and extract key terms"""
        query_lower = query.lower()
        
        for pattern_type, patterns in self.qa_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    key_term = match.group(1) if match.groups() else ""
                    return pattern_type, key_term
        
        # Default to general search
        return 'general_search', query
    
    def search_restaurants(self, query: str, limit: int = 5) -> List[Dict]:
        """Search restaurants based on query"""
        query_type, key_term = self.classify_query(query)
        
        if query_type == 'restaurant_search' and key_term:
            # Search by restaurant name
            mask = self.df['Restaurant_Name'].str.contains(key_term, case=False, na=False)
            results = self.df[mask]
        elif query_type == 'cuisine_search' and key_term:
            # Search by cuisine
            mask = self.df['Cuisines'].str.contains(key_term, case=False, na=False)
            results = self.df[mask]
        elif query_type == 'location_search' and key_term:
            # Search by location
            location_mask = (
                self.df['Location'].str.contains(key_term, case=False, na=False) |
                self.df['City'].str.contains(key_term, case=False, na=False) |
                self.df['Address'].str.contains(key_term, case=False, na=False)
            )
            results = self.df[location_mask]
        elif query_type == 'rating_search':
            # Search by rating
            results = self.df.nlargest(limit, 'Rating')
        elif query_type == 'price_search':
            # Search by price
            if 'cheap' in query_lower or 'budget' in query_lower:
                results = self.df[self.df['Price'] <= 500].nsmallest(limit, 'Price')
            elif 'expensive' in query_lower or 'fine' in query_lower:
                results = self.df[self.df['Price'] >= 1000].nlargest(limit, 'Price')
            else:
                results = self.df.nlargest(limit, 'Rating')
        elif query_type == 'feature_search':
            # Search by features
            if 'online' in query_lower:
                results = self.df[self.df['Online_Order'] == 'Yes']
            elif 'book' in query_lower:
                results = self.df[self.df['Book_Table'] == 'Yes']
            else:
                results = self.df.nlargest(limit, 'Rating')
        else:
            # General search using vector similarity
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.restaurant_vectors)[0]
            top_indices = np.argsort(similarities)[-limit:][::-1]
            results = self.df.iloc[top_indices]
        
        # Format results
        restaurants = []
        for _, row in results.head(limit).iterrows():
            restaurant = {
                'name': row['Restaurant_Name'],
                'cuisines': row['Cuisines'],
                'city': row['City'],
                'location': row['Location'],
                'address': row['Address'],
                'rating': float(row['Rating']),
                'votes': int(row['Votes']),
                'price': float(row['Price']),
                'price_range': row['Price_Range'],
                'type': row['Restaurant_Type'],
                'online_order': row['Online_Order'],
                'book_table': row['Book_Table'],
                'dish_liked': row['Dish_Liked']
            }
            restaurants.append(restaurant)
        
        return restaurants
    
    def generate_response(self, query: str) -> Dict:
        """Generate response based on query"""
        logger.info(f"Processing query: {query}")
        
        # Search for relevant restaurants
        restaurants = self.search_restaurants(query, limit=5)
        
        if not restaurants:
            return {
                'response': "I couldn't find any restaurants matching your query. Please try a different search term or ask about cuisines, locations, or restaurant features.",
                'restaurants': [],
                'query_type': 'no_results'
            }
        
        # Generate contextual response
        query_type, key_term = self.classify_query(query)
        
        if query_type == 'restaurant_search':
            response = f"I found information about {restaurants[0]['name']}. "
            response += f"It's located in {restaurants[0]['location']}, {restaurants[0]['city']}. "
            response += f"The restaurant serves {restaurants[0]['cuisines']} cuisine and has a rating of {restaurants[0]['rating']} stars. "
            response += f"Average cost for two is {restaurants[0]['price_range']}."
            
        elif query_type == 'cuisine_search':
            response = f"I found {len(restaurants)} restaurants serving {key_term} cuisine. "
            response += f"Here are the top-rated ones: {', '.join([r['name'] for r in restaurants[:3]])}. "
            response += f"The highest rated is {restaurants[0]['name']} with {restaurants[0]['rating']} stars."
            
        elif query_type == 'location_search':
            response = f"I found {len(restaurants)} restaurants in {key_term}. "
            response += f"The top-rated restaurants in this area are: {', '.join([r['name'] for r in restaurants[:3]])}. "
            response += f"{restaurants[0]['name']} has the highest rating of {restaurants[0]['rating']} stars."
            
        elif query_type == 'rating_search':
            response = f"The highest-rated restaurants in our database are: {', '.join([r['name'] for r in restaurants[:3]])}. "
            response += f"{restaurants[0]['name']} has the highest rating of {restaurants[0]['rating']} stars with {restaurants[0]['votes']} votes."
            
        elif query_type == 'price_search':
            if 'cheap' in query.lower() or 'budget' in query.lower():
                response = f"I found {len(restaurants)} budget-friendly restaurants. "
                response += f"The most affordable options are: {', '.join([r['name'] for r in restaurants[:3]])}. "
                response += f"{restaurants[0]['name']} offers great value at {restaurants[0]['price_range']}."
            else:
                response = f"I found {len(restaurants)} premium restaurants. "
                response += f"The top fine dining options are: {', '.join([r['name'] for r in restaurants[:3]])}. "
                response += f"{restaurants[0]['name']} is a premium choice with {restaurants[0]['rating']} stars."
                
        elif query_type == 'feature_search':
            if 'online' in query.lower():
                response = f"I found {len(restaurants)} restaurants that offer online ordering. "
                response += f"These include: {', '.join([r['name'] for r in restaurants[:3]])}. "
                response += f"You can order online from {restaurants[0]['name']}."
            elif 'book' in query.lower():
                response = f"I found {len(restaurants)} restaurants that accept table bookings. "
                response += f"These include: {', '.join([r['name'] for r in restaurants[:3]])}. "
                response += f"You can book a table at {restaurants[0]['name']}."
            else:
                response = f"I found {len(restaurants)} restaurants with special features. "
                response += f"Here are some options: {', '.join([r['name'] for r in restaurants[:3]])}."
                
        else:
            # General search response
            response = f"I found {len(restaurants)} restaurants matching your query. "
            response += f"The top results are: {', '.join([r['name'] for r in restaurants[:3]])}. "
            response += f"{restaurants[0]['name']} is highly rated with {restaurants[0]['rating']} stars."
        
        return {
            'response': response,
            'restaurants': restaurants,
            'query_type': query_type,
            'key_term': key_term
        }
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics for chatbot responses"""
        if self.df is None:
            return {}
        
        stats = {
            'total_restaurants': len(self.df),
            'cities': self.df['City'].nunique(),
            'cuisines': len(set([c for cuisine_list in self.df['Cuisines'].str.split(',') for c in cuisine_list])),
            'avg_rating': self.df['Rating'].mean(),
            'avg_price': self.df['Price'].mean(),
            'top_cities': self.df['City'].value_counts().head(5).to_dict(),
            'top_cuisines': self.df['Cuisines'].value_counts().head(10).to_dict(),
            'rating_distribution': self.df['Rating_Category'].value_counts().to_dict(),
            'price_distribution': self.df['Price_Category'].value_counts().to_dict()
        }
        
        return stats
    
    def answer_general_questions(self, query: str) -> str:
        """Answer general questions about the dataset"""
        query_lower = query.lower()
        stats = self.get_statistics()
        
        if 'how many' in query_lower and 'restaurant' in query_lower:
            return f"We have {stats['total_restaurants']} restaurants in our database."
        
        elif 'cities' in query_lower:
            return f"Our database covers {stats['cities']} cities. The top cities are: {', '.join(list(stats['top_cities'].keys())[:3])}."
        
        elif 'cuisines' in query_lower:
            return f"We have {stats['cuisines']} different cuisine types. Popular cuisines include: {', '.join(list(stats['top_cuisines'].keys())[:5])}."
        
        elif 'average' in query_lower and 'rating' in query_lower:
            return f"The average restaurant rating in our database is {stats['avg_rating']:.1f} stars."
        
        elif 'average' in query_lower and 'price' in query_lower:
            return f"The average cost for two people is ₹{stats['avg_price']:.0f}."
        
        elif 'best' in query_lower and 'restaurant' in query_lower:
            top_restaurant = self.df.nlargest(1, 'Rating').iloc[0]
            return f"The highest-rated restaurant is {top_restaurant['Restaurant_Name']} with {top_restaurant['Rating']} stars, located in {top_restaurant['City']}."
        
        else:
            return "I can help you find restaurants, answer questions about cuisines, locations, ratings, and prices. What would you like to know?"
    
    def initialize(self, csv_path: str) -> None:
        """Initialize the chatbot with data"""
        logger.info("Initializing restaurant chatbot...")
        
        self.load_data(csv_path)
        self.create_vector_indexes()
        self.create_qa_patterns()
        self.initialized = True
        
        logger.info("Restaurant chatbot initialized successfully")
    
    def chat(self, query: str) -> Dict:
        """Main chat function"""
        if not self.initialized:
            return {
                'response': "Chatbot is not initialized. Please load data first.",
                'restaurants': [],
                'query_type': 'error'
            }
        
        # Handle general questions
        if any(word in query.lower() for word in ['how many', 'cities', 'cuisines', 'average', 'best']):
            response = self.answer_general_questions(query)
            return {
                'response': response,
                'restaurants': [],
                'query_type': 'general_info'
            }
        
        # Handle restaurant-specific queries
        return self.generate_response(query)

def main():
    """Test the chatbot"""
    chatbot = RestaurantChatbot()
    
    try:
        # Initialize chatbot
        chatbot.initialize('data/restaurants_processed.csv')
        
        # Test queries
        test_queries = [
            "Find restaurants serving Italian cuisine",
            "Best restaurants in Bangalore",
            "Restaurants with high ratings",
            "Cheap restaurants near me",
            "How many restaurants do you have?",
            "What cuisines are available?",
            "Tell me about Empire Restaurant"
        ]
        
        print("Testing Restaurant Chatbot...")
        print("=" * 50)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response = chatbot.chat(query)
            print(f"Response: {response['response']}")
            if response['restaurants']:
                print(f"Found {len(response['restaurants'])} restaurants")
                for i, restaurant in enumerate(response['restaurants'][:2]):
                    print(f"  {i+1}. {restaurant['name']} - {restaurant['rating']} stars")
        
        print("\n[OK] Chatbot testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Chatbot testing failed: {e}")
        raise

if __name__ == "__main__":
    main()
