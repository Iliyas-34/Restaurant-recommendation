"""
Advanced Machine Learning Recommendation Engine for Restaurant Recommendations
Combines Content-Based Filtering, Collaborative Filtering, and Context-Aware Features
"""

import pandas as pd
import numpy as np
import joblib
import re
import os
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Advanced ML Libraries
try:
    from lightfm import LightFM
    from lightfm.data import Dataset
    from lightfm.evaluation import precision_at_k
    LIGHTFM_AVAILABLE = True
except ImportError:
    LIGHTFM_AVAILABLE = False
    print("LightFM not available. Collaborative filtering will be disabled.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("SentenceTransformers not available. Context-aware features will be limited.")

# Fallback for when advanced libraries are not available
if not SENTENCE_TRANSFORMERS_AVAILABLE:
    # Simple embedding fallback using TF-IDF
    class SimpleEmbedding:
        def __init__(self):
            self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            self.embeddings = None
        
        def encode(self, texts):
            if self.embeddings is None:
                self.embeddings = self.vectorizer.fit_transform(texts)
            return self.embeddings.toarray()
    
    SentenceTransformer = SimpleEmbedding
    SENTENCE_TRANSFORMERS_AVAILABLE = True

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RestaurantRecommendationEngine:
    """
    Advanced ML-based restaurant recommendation engine that combines:
    1. Content-Based Filtering (TF-IDF + Cosine Similarity)
    2. Collaborative Filtering (LightFM)
    3. Context-Aware Features (SentenceTransformers + Mood/Time/Occasion)
    """
    
    def __init__(self, data_path: str = "zomato.csv"):
        self.data_path = data_path
        self.df = None
        self.restaurants = []
        
        # ML Models
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_similarity_matrix = None
        
        # Collaborative Filtering
        self.lightfm_model = None
        self.lightfm_dataset = None
        self.user_item_matrix = None
        
        # Context-Aware Features
        self.sentence_model = None
        self.mood_embeddings = None
        self.time_embeddings = None
        self.occasion_embeddings = None
        
        # Preprocessing
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Context mappings
        self.mood_mappings = {
            'happy': ['cheerful', 'joyful', 'bright', 'vibrant', 'energetic', 'fun', 'party', 'celebration'],
            'sad': ['comfort', 'warm', 'cozy', 'healing', 'soul', 'nourishing', 'homely'],
            'angry': ['spicy', 'hot', 'intense', 'bold', 'strong', 'powerful', 'fiery'],
            'relaxed': ['calm', 'peaceful', 'serene', 'quiet', 'gentle', 'soft', 'meditation'],
            'excited': ['adventurous', 'bold', 'new', 'unique', 'thrilling', 'exciting', 'daring'],
            'bored': ['variety', 'diverse', 'different', 'new', 'interesting', 'unusual', 'creative']
        }
        
        self.time_mappings = {
            'morning': ['breakfast', 'coffee', 'brunch', 'fresh', 'light', 'healthy', 'energizing'],
            'afternoon': ['lunch', 'quick', 'efficient', 'business', 'productive', 'satisfying'],
            'evening': ['dinner', 'romantic', 'intimate', 'fine dining', 'elegant', 'sophisticated'],
            'night': ['late night', 'snacks', 'comfort', 'casual', 'relaxed', 'informal']
        }
        
        self.occasion_mappings = {
            'birthday': ['celebration', 'special', 'party', 'festive', 'sweet', 'dessert', 'cake'],
            'date': ['romantic', 'intimate', 'quiet', 'elegant', 'fine dining', 'wine', 'candlelight'],
            'party': ['group', 'fun', 'vibrant', 'loud', 'energetic', 'celebration', 'festive'],
            'lunch': ['business', 'quick', 'efficient', 'productive', 'healthy', 'light'],
            'dinner': ['family', 'intimate', 'elaborate', 'multi-course', 'fine dining'],
            'meeting': ['quiet', 'professional', 'business', 'formal', 'conference'],
            'anniversary': ['romantic', 'special', 'elegant', 'fine dining', 'wine', 'intimate']
        }
        
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess the zomato.csv dataset"""
        logger.info("Loading and preprocessing dataset...")
        
        try:
            # Load the dataset
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.df)} restaurants from dataset")
            
            # Basic data cleaning
            self.df = self.df.dropna(subset=['name', 'cuisines'])
            
            # Clean text data
            self.df['name'] = self.df['name'].astype(str).str.strip()
            self.df['cuisines'] = self.df['cuisines'].astype(str).str.strip()
            self.df['location'] = self.df['location'].astype(str).str.strip()
            self.df['rest_type'] = self.df['rest_type'].astype(str).str.strip()
            self.df['dish_liked'] = self.df['dish_liked'].fillna('').astype(str)
            self.df['reviews_list'] = self.df['reviews_list'].fillna('').astype(str)
            
            # Clean and convert numeric columns
            self.df['rate'] = self.df['rate'].str.replace('/5', '').str.strip()
            self.df['rate'] = pd.to_numeric(self.df['rate'], errors='coerce').fillna(0)
            self.df['votes'] = pd.to_numeric(self.df['votes'], errors='coerce').fillna(0)
            self.df['approx_cost(for two people)'] = pd.to_numeric(
                self.df['approx_cost(for two people)'], errors='coerce'
            ).fillna(0)
            
            # Create combined text features for ML
            self.df['combined_features'] = (
                self.df['name'] + ' ' +
                self.df['cuisines'] + ' ' +
                self.df['dish_liked'] + ' ' +
                self.df['rest_type'] + ' ' +
                self.df['location']
            )
            
            # Clean reviews and add to features
            self.df['reviews_clean'] = self.df['reviews_list'].apply(self._clean_reviews)
            self.df['combined_features'] += ' ' + self.df['reviews_clean']
            
            # Encode categorical variables
            categorical_columns = ['location', 'rest_type', 'cuisines']
            for col in categorical_columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
            
            # Create restaurant objects for easy access
            self.restaurants = self.df.to_dict('records')
            
            logger.info("Data preprocessing completed successfully")
            return self.df
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise
    
    def _clean_reviews(self, reviews_text: str) -> str:
        """Clean and preprocess review text"""
        if pd.isna(reviews_text) or reviews_text == '':
            return ''
        
        # Extract text from review tuples
        reviews = re.findall(r"'RATED\\n\s+(.*?)'", reviews_text)
        if not reviews:
            return ''
        
        # Join all reviews
        combined_reviews = ' '.join(reviews)
        
        # Clean text
        combined_reviews = re.sub(r'[^\w\s]', ' ', combined_reviews)
        combined_reviews = re.sub(r'\s+', ' ', combined_reviews)
        combined_reviews = combined_reviews.lower().strip()
        
        return combined_reviews
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for ML models"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def train_content_based_model(self):
        """Train content-based filtering model using TF-IDF"""
        logger.info("Training content-based filtering model...")
        
        try:
            # Preprocess text features
            processed_features = self.df['combined_features'].apply(self._preprocess_text)
            
            # Create TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )
            
            # Fit and transform
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_features)
            
            # Calculate cosine similarity matrix
            self.cosine_similarity_matrix = cosine_similarity(self.tfidf_matrix)
            
            logger.info("Content-based model training completed")
            
        except Exception as e:
            logger.error(f"Error training content-based model: {e}")
            raise
    
    def train_collaborative_filtering_model(self):
        """Train collaborative filtering model using LightFM"""
        if not LIGHTFM_AVAILABLE:
            logger.warning("LightFM not available. Skipping collaborative filtering.")
            return
        
        logger.info("Training collaborative filtering model...")
        
        try:
            # Create user-item interaction matrix
            # For this demo, we'll simulate user interactions based on ratings
            interactions = []
            
            for idx, row in self.df.iterrows():
                # Simulate user interactions based on rating
                rating = row['rate']
                votes = row['votes']
                
                # Create multiple user interactions based on rating and votes
                num_interactions = max(1, int(rating * 2) + int(votes / 100))
                
                for i in range(min(num_interactions, 10)):  # Limit to 10 interactions per restaurant
                    user_id = f"user_{i % 100}"  # Simulate 100 users
                    interactions.append((user_id, idx, 1.0))  # Binary interaction
            
            # Create LightFM dataset
            self.lightfm_dataset = Dataset()
            self.lightfm_dataset.fit(
                users=[f"user_{i}" for i in range(100)],
                items=[str(i) for i in range(len(self.df))]
            )
            
            # Build interaction matrix
            (interactions_matrix, weights) = self.lightfm_dataset.build_interactions(interactions)
            
            # Train LightFM model
            self.lightfm_model = LightFM(
                no_components=50,
                learning_rate=0.05,
                loss='warp',
                random_state=42
            )
            
            self.lightfm_model.fit(
                interactions_matrix,
                sample_weight=weights,
                epochs=30,
                num_threads=4,
                verbose=True
            )
            
            logger.info("Collaborative filtering model training completed")
            
        except Exception as e:
            logger.error(f"Error training collaborative filtering model: {e}")
            # Don't raise, just log the error
            logger.warning("Continuing without collaborative filtering...")
    
    def train_context_aware_model(self):
        """Train context-aware model using SentenceTransformers or fallback"""
        logger.info("Training context-aware model...")
        
        try:
            # Initialize sentence transformer (or fallback)
            if hasattr(SentenceTransformer, '__call__'):
                # Real SentenceTransformer
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            else:
                # Fallback SimpleEmbedding
                self.sentence_model = SentenceTransformer()
            
            # Create embeddings for mood, time, and occasion contexts
            self.mood_embeddings = {}
            for mood, keywords in self.mood_mappings.items():
                mood_text = ' '.join(keywords)
                embedding = self.sentence_model.encode([mood_text])
                if hasattr(embedding, 'shape') and len(embedding.shape) > 1:
                    self.mood_embeddings[mood] = embedding[0]
                else:
                    self.mood_embeddings[mood] = embedding
            
            self.time_embeddings = {}
            for time, keywords in self.time_mappings.items():
                time_text = ' '.join(keywords)
                embedding = self.sentence_model.encode([time_text])
                if hasattr(embedding, 'shape') and len(embedding.shape) > 1:
                    self.time_embeddings[time] = embedding[0]
                else:
                    self.time_embeddings[time] = embedding
            
            self.occasion_embeddings = {}
            for occasion, keywords in self.occasion_mappings.items():
                occasion_text = ' '.join(keywords)
                embedding = self.sentence_model.encode([occasion_text])
                if hasattr(embedding, 'shape') and len(embedding.shape) > 1:
                    self.occasion_embeddings[occasion] = embedding[0]
                else:
                    self.occasion_embeddings[occasion] = embedding
            
            logger.info("Context-aware model training completed")
            
        except Exception as e:
            logger.error(f"Error training context-aware model: {e}")
            # Don't raise, just log the error
            logger.warning("Continuing without advanced context features...")
    
    def get_content_based_recommendations(self, query: str, n: int = 10) -> List[Dict]:
        """Get content-based recommendations using TF-IDF and cosine similarity"""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []
        
        try:
            # Preprocess query
            processed_query = self._preprocess_text(query)
            
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([processed_query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            # Get top recommendations
            top_indices = similarities.argsort()[-n:][::-1]
            
            recommendations = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Threshold for meaningful similarity
                    restaurant = self.restaurants[idx]
                    recommendations.append({
                        'name': restaurant['name'],
                        'cuisines': restaurant['cuisines'],
                        'location': restaurant['location'],
                        'rating': restaurant['rate'],
                        'cost': restaurant['approx_cost(for two people)'],
                        'match_score': float(similarities[idx]),
                        'type': 'content_based'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
            return []
    
    def get_collaborative_recommendations(self, user_id: str = "user_0", n: int = 10) -> List[Dict]:
        """Get collaborative filtering recommendations using LightFM"""
        if not LIGHTFM_AVAILABLE or self.lightfm_model is None:
            return []
        
        try:
            # Get user recommendations
            user_idx = self.lightfm_dataset.mapping()[0][user_id]
            scores = self.lightfm_model.predict(user_idx, np.arange(len(self.df)))
            
            # Get top recommendations
            top_indices = scores.argsort()[-n:][::-1]
            
            recommendations = []
            for idx in top_indices:
                if scores[idx] > 0.1:  # Threshold for meaningful score
                    restaurant = self.restaurants[idx]
                    recommendations.append({
                        'name': restaurant['name'],
                        'cuisines': restaurant['cuisines'],
                        'location': restaurant['location'],
                        'rating': restaurant['rate'],
                        'cost': restaurant['approx_cost(for two people)'],
                        'match_score': float(scores[idx]),
                        'type': 'collaborative'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in collaborative recommendations: {e}")
            return []
    
    def get_context_aware_recommendations(self, mood: str, time: str, occasion: str, n: int = 10) -> List[Dict]:
        """Get context-aware recommendations using mood, time, and occasion"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.sentence_model is None:
            return []
        
        try:
            # Get context embeddings
            mood_embedding = self.mood_embeddings.get(mood, np.zeros(384))
            time_embedding = self.time_embeddings.get(time, np.zeros(384))
            occasion_embedding = self.occasion_embeddings.get(occasion, np.zeros(384))
            
            # Combine context embeddings
            context_embedding = (mood_embedding + time_embedding + occasion_embedding) / 3
            
            # Calculate similarities with restaurant features
            restaurant_features = []
            for restaurant in self.restaurants:
                feature_text = f"{restaurant['name']} {restaurant['cuisines']} {restaurant['rest_type']}"
                feature_embedding = self.sentence_model.encode([feature_text])[0]
                restaurant_features.append(feature_embedding)
            
            restaurant_features = np.array(restaurant_features)
            similarities = cosine_similarity([context_embedding], restaurant_features)[0]
            
            # Get top recommendations
            top_indices = similarities.argsort()[-n:][::-1]
            
            recommendations = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Threshold for meaningful similarity
                    restaurant = self.restaurants[idx]
                    recommendations.append({
                        'name': restaurant['name'],
                        'cuisines': restaurant['cuisines'],
                        'location': restaurant['location'],
                        'rating': restaurant['rate'],
                        'cost': restaurant['approx_cost(for two people)'],
                        'match_score': float(similarities[idx]),
                        'type': 'context_aware'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in context-aware recommendations: {e}")
            return []
    
    def get_hybrid_recommendations(self, query: str, mood: str, time: str, occasion: str, n: int = 10) -> List[Dict]:
        """Get hybrid recommendations combining all models"""
        logger.info(f"Getting hybrid recommendations for: {query}, mood: {mood}, time: {time}, occasion: {occasion}")
        
        # Get recommendations from each model
        content_recs = self.get_content_based_recommendations(query, n * 2)
        collab_recs = self.get_collaborative_recommendations("user_0", n * 2)
        context_recs = self.get_context_aware_recommendations(mood, time, occasion, n * 2)
        
        # Combine and score recommendations
        restaurant_scores = defaultdict(lambda: {'scores': [], 'count': 0, 'restaurant': None})
        
        # Process content-based recommendations
        for rec in content_recs:
            name = rec['name']
            restaurant_scores[name]['scores'].append(rec['match_score'] * 0.4)  # Weight: 40%
            restaurant_scores[name]['count'] += 1
            restaurant_scores[name]['restaurant'] = rec
        
        # Process collaborative recommendations
        for rec in collab_recs:
            name = rec['name']
            restaurant_scores[name]['scores'].append(rec['match_score'] * 0.3)  # Weight: 30%
            restaurant_scores[name]['count'] += 1
            if restaurant_scores[name]['restaurant'] is None:
                restaurant_scores[name]['restaurant'] = rec
        
        # Process context-aware recommendations
        for rec in context_recs:
            name = rec['name']
            restaurant_scores[name]['scores'].append(rec['match_score'] * 0.3)  # Weight: 30%
            restaurant_scores[name]['count'] += 1
            if restaurant_scores[name]['restaurant'] is None:
                restaurant_scores[name]['restaurant'] = rec
        
        # Calculate final scores
        final_recommendations = []
        for name, data in restaurant_scores.items():
            if data['restaurant'] is not None:
                # Calculate weighted average score
                avg_score = np.mean(data['scores'])
                
                # Boost score for restaurants that appear in multiple models
                diversity_boost = min(0.2, data['count'] * 0.05)
                final_score = avg_score + diversity_boost
                
                restaurant = data['restaurant'].copy()
                restaurant['match_score'] = min(1.0, final_score)  # Cap at 1.0
                restaurant['type'] = 'hybrid'
                final_recommendations.append(restaurant)
        
        # Sort by final score and return top N
        final_recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        return final_recommendations[:n]
    
    def train_all_models(self):
        """Train all ML models"""
        logger.info("Starting training of all ML models...")
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Train individual models
        self.train_content_based_model()
        self.train_collaborative_filtering_model()
        self.train_context_aware_model()
        
        logger.info("All models trained successfully!")
    
    def save_models(self, filepath: str = "restaurant_predictor.joblib"):
        """Save all trained models"""
        logger.info(f"Saving models to {filepath}")
        
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'cosine_similarity_matrix': self.cosine_similarity_matrix,
            'lightfm_model': self.lightfm_model,
            'lightfm_dataset': self.lightfm_dataset,
            'sentence_model': self.sentence_model,
            'mood_embeddings': self.mood_embeddings,
            'time_embeddings': self.time_embeddings,
            'occasion_embeddings': self.occasion_embeddings,
            'label_encoders': self.label_encoders,
            'restaurants': self.restaurants,
            'df': self.df
        }
        
        joblib.dump(model_data, filepath)
        logger.info("Models saved successfully!")
    
    def load_models(self, filepath: str = "restaurant_predictor.joblib"):
        """Load pre-trained models"""
        logger.info(f"Loading models from {filepath}")
        
        if not os.path.exists(filepath):
            logger.warning(f"Model file {filepath} not found. Training new models...")
            self.train_all_models()
            return
        
        try:
            model_data = joblib.load(filepath)
            
            self.tfidf_vectorizer = model_data.get('tfidf_vectorizer')
            self.tfidf_matrix = model_data.get('tfidf_matrix')
            self.cosine_similarity_matrix = model_data.get('cosine_similarity_matrix')
            self.lightfm_model = model_data.get('lightfm_model')
            self.lightfm_dataset = model_data.get('lightfm_dataset')
            self.sentence_model = model_data.get('sentence_model')
            self.mood_embeddings = model_data.get('mood_embeddings', {})
            self.time_embeddings = model_data.get('time_embeddings', {})
            self.occasion_embeddings = model_data.get('occasion_embeddings', {})
            self.label_encoders = model_data.get('label_encoders', {})
            self.restaurants = model_data.get('restaurants', [])
            self.df = model_data.get('df')
            
            logger.info("Models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.warning("Training new models...")
            self.train_all_models()

# Global recommendation engine instance
recommendation_engine = RestaurantRecommendationEngine()

def get_recommendations(user_input: str, mood: str, time: str, occasion: str, n: int = 5) -> List[Dict]:
    """Get restaurant recommendations using the hybrid ML engine"""
    return recommendation_engine.get_hybrid_recommendations(user_input, mood, time, occasion, n)

def initialize_recommendation_engine():
    """Initialize the recommendation engine"""
    global recommendation_engine
    recommendation_engine.load_models()
    logger.info("Recommendation engine initialized successfully!")
