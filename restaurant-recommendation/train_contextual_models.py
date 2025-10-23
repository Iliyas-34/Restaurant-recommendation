#!/usr/bin/env python3
"""
Lightweight ML Models for Mood/Time/Occasion Personalization
Trains specialized models for contextual restaurant recommendations
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextualRecommendationEngine:
    """Lightweight ML engine for contextual restaurant recommendations"""
    
    def __init__(self):
        self.mood_model = None
        self.time_model = None
        self.occasion_model = None
        self.cuisine_vectorizer = None
        self.restaurant_vectorizer = None
        self.cuisine_similarity = None
        self.restaurant_similarity = None
        self.label_encoders = {}
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and prepare data"""
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} restaurants")
        return df
    
    def create_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for contextual recommendations"""
        logger.info("Creating contextual features...")
        
        # Create cuisine embeddings
        df['cuisine_features'] = df['Cuisines'].fillna('Unknown')
        
        # Create restaurant embeddings
        df['restaurant_features'] = df.apply(
            lambda row: f"{row['Restaurant_Name']} {row['Cuisines']} {row['Location']} {row['City']}", 
            axis=1
        )
        
        # Create price categories
        df['price_category'] = pd.cut(
            df['Price'], 
            bins=[0, 300, 600, 1200, float('inf')], 
            labels=['Budget', 'Moderate', 'Expensive', 'Very Expensive']
        )
        
        # Create rating categories
        df['rating_category'] = pd.cut(
            df['Rating'], 
            bins=[0, 3.0, 3.5, 4.0, 4.5, 5.0], 
            labels=['Poor', 'Average', 'Good', 'Very Good', 'Excellent']
        )
        
        return df
    
    def train_cuisine_similarity(self, df: pd.DataFrame) -> None:
        """Train cuisine similarity model"""
        logger.info("Training cuisine similarity model...")
        
        # Create TF-IDF vectors for cuisines
        self.cuisine_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        cuisine_matrix = self.cuisine_vectorizer.fit_transform(df['cuisine_features'])
        self.cuisine_similarity = cosine_similarity(cuisine_matrix)
        
        logger.info("Cuisine similarity model trained")
    
    def train_restaurant_similarity(self, df: pd.DataFrame) -> None:
        """Train restaurant similarity model"""
        logger.info("Training restaurant similarity model...")
        
        # Create TF-IDF vectors for restaurants
        self.restaurant_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        restaurant_matrix = self.restaurant_vectorizer.fit_transform(df['restaurant_features'])
        self.restaurant_similarity = cosine_similarity(restaurant_matrix)
        
        logger.info("Restaurant similarity model trained")
    
    def create_mood_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create training data for mood-based recommendations"""
        logger.info("Creating mood training data...")
        
        # Define mood-cuisine mappings
        mood_cuisine_mappings = {
            'happy': ['Desserts', 'Fast Food', 'Street Food', 'Beverages', 'Bakery', 'Cafe'],
            'sad': ['Comfort Food', 'Desserts', 'Fast Food', 'Beverages'],
            'excited': ['Fast Food', 'Street Food', 'BBQ', 'Seafood', 'Fusion'],
            'relaxed': ['Cafe', 'Beverages', 'Mediterranean', 'Continental', 'Fine Dining'],
            'angry': ['Spicy Food', 'BBQ', 'Fast Food', 'Street Food'],
            'bored': ['Fusion', 'New Cuisine', 'Experimental', 'Fine Dining']
        }
        
        # Create mood labels based on cuisine preferences
        mood_labels = []
        features = []
        
        for _, row in df.iterrows():
            cuisines = str(row['Cuisines']).lower()
            restaurant_features = [
                row['Rating'],
                row['Price'],
                row['Votes'],
                1 if 'online' in str(row['Online_Order']).lower() else 0,
                1 if 'yes' in str(row['Book_Table']).lower() else 0
            ]
            
            # Determine mood based on cuisine
            mood_scores = {}
            for mood, preferred_cuisines in mood_cuisine_mappings.items():
                score = sum(1 for cuisine in preferred_cuisines if cuisine.lower() in cuisines)
                mood_scores[mood] = score
            
            # Assign mood based on highest score
            if max(mood_scores.values()) > 0:
                predicted_mood = max(mood_scores, key=mood_scores.get)
            else:
                predicted_mood = 'neutral'
            
            mood_labels.append(predicted_mood)
            features.append(restaurant_features)
        
        return np.array(features), np.array(mood_labels)
    
    def create_time_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create training data for time-based recommendations"""
        logger.info("Creating time training data...")
        
        # Define time-cuisine mappings
        time_cuisine_mappings = {
            'morning': ['Breakfast', 'Cafe', 'Bakery', 'Beverages'],
            'afternoon': ['Lunch', 'Fast Food', 'Street Food', 'Cafe'],
            'evening': ['Dinner', 'Fine Dining', 'BBQ', 'Seafood'],
            'night': ['Late Night', 'Fast Food', 'Street Food', 'Beverages']
        }
        
        time_labels = []
        features = []
        
        for _, row in df.iterrows():
            cuisines = str(row['Cuisines']).lower()
            restaurant_features = [
                row['Rating'],
                row['Price'],
                row['Votes'],
                1 if 'online' in str(row['Online_Order']).lower() else 0,
                1 if 'yes' in str(row['Book_Table']).lower() else 0
            ]
            
            # Determine time based on cuisine and restaurant type
            time_scores = {}
            for time, preferred_cuisines in time_cuisine_mappings.items():
                score = sum(1 for cuisine in preferred_cuisines if cuisine.lower() in cuisines)
                time_scores[time] = score
            
            # Assign time based on highest score
            if max(time_scores.values()) > 0:
                predicted_time = max(time_scores, key=time_scores.get)
            else:
                predicted_time = 'anytime'
            
            time_labels.append(predicted_time)
            features.append(restaurant_features)
        
        return np.array(features), np.array(time_labels)
    
    def create_occasion_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create training data for occasion-based recommendations"""
        logger.info("Creating occasion training data...")
        
        # Define occasion-cuisine mappings
        occasion_cuisine_mappings = {
            'date': ['Fine Dining', 'Mediterranean', 'Continental', 'Romantic'],
            'birthday': ['Desserts', 'Fine Dining', 'Party Food', 'Celebration'],
            'party': ['Party Food', 'BBQ', 'Finger Food', 'Beverages'],
            'lunch': ['Lunch', 'Fast Food', 'Cafe', 'Quick Bites'],
            'dinner': ['Dinner', 'Fine Dining', 'BBQ', 'Seafood'],
            'meeting': ['Cafe', 'Business Lunch', 'Quiet Dining', 'Professional'],
            'anniversary': ['Fine Dining', 'Romantic', 'Special Occasion', 'Celebration']
        }
        
        occasion_labels = []
        features = []
        
        for _, row in df.iterrows():
            cuisines = str(row['Cuisines']).lower()
            restaurant_features = [
                row['Rating'],
                row['Price'],
                row['Votes'],
                1 if 'online' in str(row['Online_Order']).lower() else 0,
                1 if 'yes' in str(row['Book_Table']).lower() else 0
            ]
            
            # Determine occasion based on cuisine and restaurant type
            occasion_scores = {}
            for occasion, preferred_cuisines in occasion_cuisine_mappings.items():
                score = sum(1 for cuisine in preferred_cuisines if cuisine.lower() in cuisines)
                occasion_scores[occasion] = score
            
            # Assign occasion based on highest score
            if max(occasion_scores.values()) > 0:
                predicted_occasion = max(occasion_scores, key=occasion_scores.get)
            else:
                predicted_occasion = 'casual'
            
            occasion_labels.append(predicted_occasion)
            features.append(restaurant_features)
        
        return np.array(features), np.array(occasion_labels)
    
    def train_contextual_models(self, df: pd.DataFrame) -> None:
        """Train all contextual models"""
        logger.info("Training contextual models...")
        
        # Train mood model
        mood_features, mood_labels = self.create_mood_training_data(df)
        self.mood_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.mood_model.fit(mood_features, mood_labels)
        logger.info("Mood model trained")
        
        # Train time model
        time_features, time_labels = self.create_time_training_data(df)
        self.time_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.time_model.fit(time_features, time_labels)
        logger.info("Time model trained")
        
        # Train occasion model
        occasion_features, occasion_labels = self.create_occasion_training_data(df)
        self.occasion_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.occasion_model.fit(occasion_features, occasion_labels)
        logger.info("Occasion model trained")
    
    def get_contextual_recommendations(self, df: pd.DataFrame, mood: str = None, 
                                     time: str = None, occasion: str = None, 
                                     limit: int = 10) -> List[Dict]:
        """Get contextual recommendations"""
        logger.info(f"Getting recommendations for mood={mood}, time={time}, occasion={occasion}")
        
        results = df.copy()
        
        # Apply mood-based filtering
        if mood and self.mood_model:
            mood_features = results[['Rating', 'Price', 'Votes', 'Online_Order', 'Book_Table']].copy()
            mood_features['Online_Order'] = mood_features['Online_Order'].apply(
                lambda x: 1 if 'yes' in str(x).lower() else 0
            )
            mood_features['Book_Table'] = mood_features['Book_Table'].apply(
                lambda x: 1 if 'yes' in str(x).lower() else 0
            )
            
            mood_predictions = self.mood_model.predict(mood_features)
            mood_probs = self.mood_model.predict_proba(mood_features)
            
            # Filter by mood
            mood_mask = mood_predictions == mood
            results = results[mood_mask]
            
            if len(results) == 0:
                logger.warning(f"No restaurants found for mood: {mood}")
                return []
        
        # Apply time-based filtering
        if time and self.time_model:
            time_features = results[['Rating', 'Price', 'Votes', 'Online_Order', 'Book_Table']].copy()
            time_features['Online_Order'] = time_features['Online_Order'].apply(
                lambda x: 1 if 'yes' in str(x).lower() else 0
            )
            time_features['Book_Table'] = time_features['Book_Table'].apply(
                lambda x: 1 if 'yes' in str(x).lower() else 0
            )
            
            time_predictions = self.time_model.predict(time_features)
            time_mask = time_predictions == time
            results = results[time_mask]
            
            if len(results) == 0:
                logger.warning(f"No restaurants found for time: {time}")
                return []
        
        # Apply occasion-based filtering
        if occasion and self.occasion_model:
            occasion_features = results[['Rating', 'Price', 'Votes', 'Online_Order', 'Book_Table']].copy()
            occasion_features['Online_Order'] = occasion_features['Online_Order'].apply(
                lambda x: 1 if 'yes' in str(x).lower() else 0
            )
            occasion_features['Book_Table'] = occasion_features['Book_Table'].apply(
                lambda x: 1 if 'yes' in str(x).lower() else 0
            )
            
            occasion_predictions = self.occasion_model.predict(occasion_features)
            occasion_mask = occasion_predictions == occasion
            results = results[occasion_mask]
            
            if len(results) == 0:
                logger.warning(f"No restaurants found for occasion: {occasion}")
                return []
        
        # Sort by rating and votes
        results = results.sort_values(['Rating', 'Votes'], ascending=[False, False])
        
        # Return top recommendations
        recommendations = []
        for _, row in results.head(limit).iterrows():
            recommendation = {
                'name': row['Restaurant_Name'],
                'cuisines': row['Cuisines'],
                'city': row['City'],
                'location': row['Location'],
                'rating': float(row['Rating']),
                'votes': int(row['Votes']),
                'price': float(row['Price']),
                'price_range': row['Price_Range'],
                'type': row['Restaurant_Type'],
                'latitude': float(row['Latitude']) if pd.notna(row['Latitude']) else None,
                'longitude': float(row['Longitude']) if pd.notna(row['Longitude']) else None,
                'online_order': row['Online_Order'],
                'book_table': row['Book_Table']
            }
            recommendations.append(recommendation)
        
        logger.info(f"Found {len(recommendations)} contextual recommendations")
        return recommendations
    
    def save_models(self, output_dir: str = 'models') -> None:
        """Save all trained models"""
        logger.info(f"Saving models to {output_dir}")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save contextual models
        with open(f'{output_dir}/mood_model.pkl', 'wb') as f:
            pickle.dump(self.mood_model, f)
        
        with open(f'{output_dir}/time_model.pkl', 'wb') as f:
            pickle.dump(self.time_model, f)
        
        with open(f'{output_dir}/occasion_model.pkl', 'wb') as f:
            pickle.dump(self.occasion_model, f)
        
        # Save vectorizers
        with open(f'{output_dir}/cuisine_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.cuisine_vectorizer, f)
        
        with open(f'{output_dir}/restaurant_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.restaurant_vectorizer, f)
        
        # Save similarity matrices
        with open(f'{output_dir}/cuisine_similarity.pkl', 'wb') as f:
            pickle.dump(self.cuisine_similarity, f)
        
        with open(f'{output_dir}/restaurant_similarity.pkl', 'wb') as f:
            pickle.dump(self.restaurant_similarity, f)
        
        logger.info("All models saved successfully")
    
    def load_models(self, model_dir: str = 'models') -> None:
        """Load all trained models"""
        logger.info(f"Loading models from {model_dir}")
        
        try:
            # Load contextual models
            with open(f'{model_dir}/mood_model.pkl', 'rb') as f:
                self.mood_model = pickle.load(f)
            
            with open(f'{model_dir}/time_model.pkl', 'rb') as f:
                self.time_model = pickle.load(f)
            
            with open(f'{model_dir}/occasion_model.pkl', 'rb') as f:
                self.occasion_model = pickle.load(f)
            
            # Load vectorizers
            with open(f'{model_dir}/cuisine_vectorizer.pkl', 'rb') as f:
                self.cuisine_vectorizer = pickle.load(f)
            
            with open(f'{model_dir}/restaurant_vectorizer.pkl', 'rb') as f:
                self.restaurant_vectorizer = pickle.load(f)
            
            # Load similarity matrices
            with open(f'{model_dir}/cuisine_similarity.pkl', 'rb') as f:
                self.cuisine_similarity = pickle.load(f)
            
            with open(f'{model_dir}/restaurant_similarity.pkl', 'rb') as f:
                self.restaurant_similarity = pickle.load(f)
            
            logger.info("All models loaded successfully")
            
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            raise

def main():
    """Main training function"""
    engine = ContextualRecommendationEngine()
    
    try:
        # Load data
        df = engine.load_data('data/restaurants_processed.csv')
        
        # Create features
        df = engine.create_contextual_features(df)
        
        # Train similarity models
        engine.train_cuisine_similarity(df)
        engine.train_restaurant_similarity(df)
        
        # Train contextual models
        engine.train_contextual_models(df)
        
        # Save models
        engine.save_models()
        
        # Test recommendations
        logger.info("Testing recommendations...")
        test_recommendations = engine.get_contextual_recommendations(
            df, mood='happy', time='evening', occasion='dinner', limit=5
        )
        
        logger.info(f"Test recommendations: {len(test_recommendations)} restaurants")
        for rec in test_recommendations[:3]:
            logger.info(f"- {rec['name']}: {rec['cuisines']} (Rating: {rec['rating']})")
        
        print("[OK] Contextual ML models trained successfully!")
        print(f"[INFO] Trained models for mood, time, and occasion recommendations")
        print(f"[INFO] Test recommendations: {len(test_recommendations)} restaurants found")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
