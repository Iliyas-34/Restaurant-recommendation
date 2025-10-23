"""
Enhanced ML Recommendation Engine for Restaurant Finder
Uses restaurants.json data for training and recommendations
"""

import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from collections import defaultdict
import re

class EnhancedRestaurantML:
    def __init__(self, restaurants_data):
        """
        Initialize the ML engine with restaurant data
        
        Args:
            restaurants_data: List of restaurant dictionaries from restaurants.json
        """
        self.restaurants = restaurants_data
        self.df = None
        self.tfidf_vectorizer = None
        self.cosine_similarity_matrix = None
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Initialize the ML models
        self._prepare_data()
        self._train_models()
        
    def _prepare_data(self):
        """Prepare and clean the restaurant data for ML training"""
        print("🔄 Preparing restaurant data for ML training...")
        
        # Convert to DataFrame
        self.df = pd.DataFrame(self.restaurants)
        
        # Clean and standardize data
        self.df['Restaurant Name'] = self.df['Restaurant Name'].fillna('Unknown')
        self.df['Cuisines'] = self.df['Cuisines'].fillna('Unknown')
        self.df['City'] = self.df['City'].fillna('Unknown')
        self.df['Location'] = self.df['Location'].fillna('Unknown')
        self.df['Restaurant Type'] = self.df['Restaurant Type'].fillna('Unknown')
        self.df['Dish Liked'] = self.df['Dish Liked'].fillna('Unknown')
        
        # Handle numeric columns
        self.df['Aggregate rating'] = pd.to_numeric(self.df['Aggregate rating'], errors='coerce').fillna(0)
        self.df['Average Cost for two'] = pd.to_numeric(self.df['Average Cost for two'], errors='coerce').fillna(0)
        self.df['Votes'] = pd.to_numeric(self.df['Votes'], errors='coerce').fillna(0)
        
        # Create enhanced features
        self._create_enhanced_features()
        
        print(f"✅ Data prepared: {len(self.df)} restaurants")
        
    def _create_enhanced_features(self):
        """Create enhanced features for better recommendations"""
        
        # Create cuisine categories
        self.df['Cuisine_Categories'] = self.df['Cuisines'].apply(self._categorize_cuisines)
        
        # Create price categories
        self.df['Price_Category'] = self.df['Average Cost for two'].apply(self._categorize_price)
        
        # Create rating categories
        self.df['Rating_Category'] = self.df['Aggregate rating'].apply(self._categorize_rating)
        
        # Create location features
        self.df['Location_Type'] = self.df['Location'].apply(self._categorize_location)
        
        # Create restaurant type features
        self.df['Restaurant_Category'] = self.df['Restaurant Type'].apply(self._categorize_restaurant_type)
        
        # Create combined text features for similarity
        self.df['Combined_Features'] = (
            self.df['Restaurant Name'].astype(str) + " " +
            self.df['Cuisines'].astype(str) + " " +
            self.df['City'].astype(str) + " " +
            self.df['Location'].astype(str) + " " +
            self.df['Restaurant Type'].astype(str) + " " +
            self.df['Dish Liked'].astype(str) + " " +
            self.df['Cuisine_Categories'].astype(str) + " " +
            self.df['Price_Category'].astype(str) + " " +
            self.df['Rating_Category'].astype(str)
        )
        
        # Create mood-based features
        self.df['Mood_Features'] = self.df.apply(self._extract_mood_features, axis=1)
        
        # Create time-based features
        self.df['Time_Features'] = self.df.apply(self._extract_time_features, axis=1)
        
        # Create occasion-based features
        self.df['Occasion_Features'] = self.df.apply(self._extract_occasion_features, axis=1)
        
    def _categorize_cuisines(self, cuisines_str):
        """Categorize cuisines into broader categories"""
        if not cuisines_str or cuisines_str == 'Unknown':
            return 'Unknown'
        
        cuisines_lower = cuisines_str.lower()
        
        # Define cuisine categories
        categories = {
            'Indian': ['indian', 'north indian', 'south indian', 'biryani', 'curry', 'tandoor', 'mughlai', 'rajasthani', 'gujarati', 'punjabi', 'bengali', 'kerala', 'tamil', 'telugu', 'kannada', 'marathi'],
            'Asian': ['chinese', 'japanese', 'thai', 'korean', 'vietnamese', 'indonesian', 'malaysian', 'singaporean', 'asian', 'dimsum', 'sushi', 'ramen', 'pad thai'],
            'European': ['italian', 'french', 'spanish', 'german', 'british', 'european', 'continental', 'mediterranean', 'greek', 'turkish'],
            'American': ['american', 'mexican', 'tex-mex', 'burger', 'pizza', 'fast food', 'bbq', 'steak'],
            'Middle Eastern': ['arabian', 'lebanese', 'persian', 'turkish', 'shawarma', 'kebab', 'falafel'],
            'Desserts': ['dessert', 'bakery', 'cake', 'ice cream', 'pastry', 'sweets', 'chocolate'],
            'Beverages': ['coffee', 'tea', 'juice', 'beverages', 'bar', 'pub', 'brewery', 'cocktail'],
            'Healthy': ['vegan', 'vegetarian', 'salad', 'healthy', 'organic', 'diet', 'protein'],
            'Street Food': ['street food', 'snacks', 'chaat', 'vada pav', 'dosa', 'idli', 'samosa']
        }
        
        for category, keywords in categories.items():
            if any(keyword in cuisines_lower for keyword in keywords):
                return category
        
        return 'Other'
    
    def _categorize_price(self, price):
        """Categorize price into ranges"""
        if pd.isna(price) or price == 0:
            return 'Unknown'
        elif price <= 500:
            return 'Budget'
        elif price <= 1000:
            return 'Moderate'
        elif price <= 2000:
            return 'Expensive'
        else:
            return 'Premium'
    
    def _categorize_rating(self, rating):
        """Categorize rating into ranges"""
        if pd.isna(rating) or rating == 0:
            return 'Unknown'
        elif rating >= 4.5:
            return 'Excellent'
        elif rating >= 4.0:
            return 'Very Good'
        elif rating >= 3.5:
            return 'Good'
        elif rating >= 3.0:
            return 'Average'
        else:
            return 'Poor'
    
    def _categorize_location(self, location):
        """Categorize location types"""
        if not location or location == 'Unknown':
            return 'Unknown'
        
        location_lower = location.lower()
        
        if any(keyword in location_lower for keyword in ['mall', 'shopping', 'center']):
            return 'Mall'
        elif any(keyword in location_lower for keyword in ['street', 'road', 'avenue']):
            return 'Street'
        elif any(keyword in location_lower for keyword in ['hotel', 'resort']):
            return 'Hotel'
        elif any(keyword in location_lower for keyword in ['airport', 'station']):
            return 'Transport'
        else:
            return 'Other'
    
    def _categorize_restaurant_type(self, restaurant_type):
        """Categorize restaurant types"""
        if not restaurant_type or restaurant_type == 'Unknown':
            return 'Unknown'
        
        type_lower = restaurant_type.lower()
        
        if any(keyword in type_lower for keyword in ['fine dining', 'upscale', 'premium']):
            return 'Fine Dining'
        elif any(keyword in type_lower for keyword in ['casual', 'family', 'dining']):
            return 'Casual Dining'
        elif any(keyword in type_lower for keyword in ['quick', 'fast', 'takeaway']):
            return 'Quick Service'
        elif any(keyword in type_lower for keyword in ['cafe', 'coffee', 'bakery']):
            return 'Cafe'
        elif any(keyword in type_lower for keyword in ['bar', 'pub', 'brewery', 'lounge']):
            return 'Bar/Pub'
        else:
            return 'Other'
    
    def _extract_mood_features(self, row):
        """Extract mood-based features from restaurant data"""
        features = []
        
        # Happy mood indicators
        if any(keyword in str(row['Dish Liked']).lower() for keyword in ['dessert', 'cake', 'ice cream', 'sweet', 'chocolate']):
            features.append('happy_desserts')
        if any(keyword in str(row['Cuisines']).lower() for keyword in ['fast food', 'burger', 'pizza', 'fried']):
            features.append('happy_comfort')
        if row['Restaurant Type'].lower() in ['pub', 'bar', 'brewery', 'lounge']:
            features.append('happy_social')
        
        # Angry mood indicators
        if any(keyword in str(row['Cuisines']).lower() for keyword in ['spicy', 'hot', 'chili', 'pepper']):
            features.append('angry_spicy')
        if any(keyword in str(row['Dish Liked']).lower() for keyword in ['spicy', 'hot', 'chili']):
            features.append('angry_spicy_food')
        
        # Sad mood indicators
        if any(keyword in str(row['Cuisines']).lower() for keyword in ['comfort', 'home', 'traditional']):
            features.append('sad_comfort')
        if any(keyword in str(row['Dish Liked']).lower() for keyword in ['soup', 'warm', 'comfort']):
            features.append('sad_comfort_food')
        
        return ' '.join(features)
    
    def _extract_time_features(self, row):
        """Extract time-based features from restaurant data"""
        features = []
        
        # Morning indicators
        if any(keyword in str(row['Cuisines']).lower() for keyword in ['breakfast', 'brunch', 'coffee', 'tea']):
            features.append('morning_breakfast')
        if any(keyword in str(row['Restaurant Type']).lower() for keyword in ['cafe', 'bakery', 'coffee']):
            features.append('morning_cafe')
        
        # Afternoon indicators
        if any(keyword in str(row['Cuisines']).lower() for keyword in ['lunch', 'quick', 'fast']):
            features.append('afternoon_lunch')
        if any(keyword in str(row['Restaurant Type']).lower() for keyword in ['casual', 'family']):
            features.append('afternoon_casual')
        
        # Evening indicators
        if any(keyword in str(row['Cuisines']).lower() for keyword in ['dinner', 'fine dining', 'upscale']):
            features.append('evening_dinner')
        if any(keyword in str(row['Restaurant Type']).lower() for keyword in ['fine dining', 'upscale', 'premium']):
            features.append('evening_fine_dining')
        
        # Night indicators
        if any(keyword in str(row['Restaurant Type']).lower() for keyword in ['bar', 'pub', 'lounge', 'nightclub']):
            features.append('night_social')
        if any(keyword in str(row['Cuisines']).lower() for keyword in ['late night', '24/7', 'night']):
            features.append('night_late')
        
        return ' '.join(features)
    
    def _extract_occasion_features(self, row):
        """Extract occasion-based features from restaurant data"""
        features = []
        
        # Birthday indicators
        if any(keyword in str(row['Dish Liked']).lower() for keyword in ['cake', 'dessert', 'celebration', 'birthday']):
            features.append('birthday_desserts')
        if any(keyword in str(row['Restaurant Type']).lower() for keyword in ['fine dining', 'upscale', 'premium']):
            features.append('birthday_fine_dining')
        
        # Family gathering indicators
        if any(keyword in str(row['Restaurant Type']).lower() for keyword in ['family', 'casual', 'dining']):
            features.append('family_casual')
        if any(keyword in str(row['Cuisines']).lower() for keyword in ['multicuisine', 'indian', 'continental']):
            features.append('family_multicuisine')
        
        # Date night indicators
        if any(keyword in str(row['Restaurant Type']).lower() for keyword in ['fine dining', 'romantic', 'upscale']):
            features.append('date_fine_dining')
        if any(keyword in str(row['Cuisines']).lower() for keyword in ['italian', 'french', 'mediterranean']):
            features.append('date_romantic')
        
        # Business meeting indicators
        if any(keyword in str(row['Restaurant Type']).lower() for keyword in ['fine dining', 'upscale', 'business']):
            features.append('business_formal')
        if any(keyword in str(row['Cuisines']).lower() for keyword in ['continental', 'european', 'international']):
            features.append('business_international')
        
        return ' '.join(features)
    
    def _train_models(self):
        """Train the ML models"""
        print("🔄 Training ML models...")
        
        # Train TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['Combined_Features'])
        self.cosine_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Train clustering model for restaurant grouping
        features_for_clustering = self._prepare_clustering_features()
        self.kmeans_model = KMeans(n_clusters=50, random_state=42, n_init=10)
        self.kmeans_model.fit(features_for_clustering)
        
        print(f"✅ ML models trained successfully!")
        print(f"   - TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"   - Cosine similarity matrix shape: {self.cosine_similarity_matrix.shape}")
        print(f"   - KMeans clusters: {self.kmeans_model.n_clusters}")
        
    def _prepare_clustering_features(self):
        """Prepare features for clustering"""
        # Create numerical features for clustering
        features = []
        
        # Rating (normalized)
        rating_norm = self.df['Aggregate rating'] / 5.0
        
        # Price (normalized)
        price_norm = self.df['Average Cost for two'] / self.df['Average Cost for two'].max()
        
        # Votes (log normalized)
        votes_norm = np.log1p(self.df['Votes']) / np.log1p(self.df['Votes']).max()
        
        # Encode categorical features
        cuisine_encoded = pd.get_dummies(self.df['Cuisine_Categories']).values
        price_cat_encoded = pd.get_dummies(self.df['Price_Category']).values
        rating_cat_encoded = pd.get_dummies(self.df['Rating_Category']).values
        
        # Combine all features
        features = np.column_stack([
            rating_norm.values,
            price_norm.values,
            votes_norm.values,
            cuisine_encoded,
            price_cat_encoded,
            rating_cat_encoded
        ])
        
        return features
    
    def get_recommendations(self, restaurant_name=None, mood=None, time=None, occasion=None, 
                          cuisine=None, city=None, limit=10):
        """
        Get restaurant recommendations based on various criteria
        
        Args:
            restaurant_name: Name of restaurant to find similar ones
            mood: Mood filter (happy, angry, sad, excited)
            time: Time filter (morning, afternoon, evening, night)
            occasion: Occasion filter (birthday, family, date, business)
            cuisine: Cuisine filter
            city: City filter
            limit: Number of recommendations to return
            
        Returns:
            List of recommended restaurants with scores
        """
        recommendations = []
        
        if restaurant_name:
            # Find similar restaurants based on content similarity
            recommendations = self._get_similar_restaurants(restaurant_name, limit)
        else:
            # Get recommendations based on filters
            recommendations = self._get_filtered_recommendations(
                mood, time, occasion, cuisine, city, limit
            )
        
        return recommendations
    
    def _get_similar_restaurants(self, restaurant_name, limit):
        """Get restaurants similar to the given restaurant"""
        # Find the restaurant index
        restaurant_idx = None
        for idx, row in self.df.iterrows():
            if restaurant_name.lower() in row['Restaurant Name'].lower():
                restaurant_idx = idx
                break
        
        if restaurant_idx is None:
            return []
        
        # Get similarity scores
        similarity_scores = self.cosine_similarity_matrix[restaurant_idx]
        
        # Get top similar restaurants (excluding the restaurant itself)
        similar_indices = np.argsort(similarity_scores)[::-1][1:limit+1]
        
        recommendations = []
        for idx in similar_indices:
            restaurant = self.df.iloc[idx]
            score = similarity_scores[idx]
            
            recommendations.append({
                'name': restaurant['Restaurant Name'],
                'cuisines': restaurant['Cuisines'],
                'city': restaurant['City'],
                'location': restaurant['Location'],
                'rating': restaurant['Aggregate rating'],
                'price': restaurant['Average Cost for two'],
                'votes': restaurant['Votes'],
                'type': restaurant['Restaurant Type'],
                'dish_liked': restaurant['Dish Liked'],
                'similarity_score': float(score),
                'match_reason': f"Similar to {restaurant_name}"
            })
        
        return recommendations
    
    def _get_filtered_recommendations(self, mood, time, occasion, cuisine, city, limit):
        """Get recommendations based on filters"""
        filtered_df = self.df.copy()
        
        # Apply filters
        if cuisine:
            filtered_df = filtered_df[filtered_df['Cuisines'].str.contains(cuisine, case=False, na=False)]
        
        if city:
            filtered_df = filtered_df[filtered_df['City'].str.contains(city, case=False, na=False)]
        
        if mood:
            mood_features = self._get_mood_keywords(mood)
            mask = filtered_df['Mood_Features'].str.contains('|'.join(mood_features), case=False, na=False)
            filtered_df = filtered_df[mask]
        
        if time:
            time_features = self._get_time_keywords(time)
            mask = filtered_df['Time_Features'].str.contains('|'.join(time_features), case=False, na=False)
            filtered_df = filtered_df[mask]
        
        if occasion:
            occasion_features = self._get_occasion_keywords(occasion)
            mask = filtered_df['Occasion_Features'].str.contains('|'.join(occasion_features), case=False, na=False)
            filtered_df = filtered_df[mask]
        
        # Sort by rating and votes
        filtered_df = filtered_df.sort_values(['Aggregate rating', 'Votes'], ascending=[False, False])
        
        # Return top recommendations
        recommendations = []
        for _, restaurant in filtered_df.head(limit).iterrows():
            recommendations.append({
                'name': restaurant['Restaurant Name'],
                'cuisines': restaurant['Cuisines'],
                'city': restaurant['City'],
                'location': restaurant['Location'],
                'rating': restaurant['Aggregate rating'],
                'price': restaurant['Average Cost for two'],
                'votes': restaurant['Votes'],
                'type': restaurant['Restaurant Type'],
                'dish_liked': restaurant['Dish Liked'],
                'similarity_score': 1.0,
                'match_reason': f"Matches your filters: {mood or 'any mood'}, {time or 'any time'}, {occasion or 'any occasion'}"
            })
        
        return recommendations
    
    def _get_mood_keywords(self, mood):
        """Get keywords for mood-based filtering"""
        mood_keywords = {
            'happy': ['happy_desserts', 'happy_comfort', 'happy_social'],
            'angry': ['angry_spicy', 'angry_spicy_food'],
            'sad': ['sad_comfort', 'sad_comfort_food'],
            'excited': ['happy_social', 'happy_comfort']
        }
        return mood_keywords.get(mood.lower(), [])
    
    def _get_time_keywords(self, time):
        """Get keywords for time-based filtering"""
        time_keywords = {
            'morning': ['morning_breakfast', 'morning_cafe'],
            'afternoon': ['afternoon_lunch', 'afternoon_casual'],
            'evening': ['evening_dinner', 'evening_fine_dining'],
            'night': ['night_social', 'night_late']
        }
        return time_keywords.get(time.lower(), [])
    
    def _get_occasion_keywords(self, occasion):
        """Get keywords for occasion-based filtering"""
        occasion_keywords = {
            'birthday': ['birthday_desserts', 'birthday_fine_dining'],
            'family': ['family_casual', 'family_multicuisine'],
            'date': ['date_fine_dining', 'date_romantic'],
            'business': ['business_formal', 'business_international']
        }
        return occasion_keywords.get(occasion.lower(), [])
    
    def save_models(self, filepath='models/'):
        """Save trained models to disk"""
        os.makedirs(filepath, exist_ok=True)
        
        # Save TF-IDF vectorizer
        with open(os.path.join(filepath, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        # Save cosine similarity matrix
        np.save(os.path.join(filepath, 'cosine_similarity_matrix.npy'), self.cosine_similarity_matrix)
        
        # Save KMeans model
        with open(os.path.join(filepath, 'kmeans_model.pkl'), 'wb') as f:
            pickle.dump(self.kmeans_model, f)
        
        # Save scaler
        with open(os.path.join(filepath, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"✅ Models saved to {filepath}")
    
    def load_models(self, filepath='models/'):
        """Load trained models from disk"""
        try:
            # Load TF-IDF vectorizer
            with open(os.path.join(filepath, 'tfidf_vectorizer.pkl'), 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            # Load cosine similarity matrix
            self.cosine_similarity_matrix = np.load(os.path.join(filepath, 'cosine_similarity_matrix.npy'))
            
            # Load KMeans model
            with open(os.path.join(filepath, 'kmeans_model.pkl'), 'rb') as f:
                self.kmeans_model = pickle.load(f)
            
            # Load scaler
            with open(os.path.join(filepath, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            
            print(f"✅ Models loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
