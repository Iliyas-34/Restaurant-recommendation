#!/usr/bin/env python3
"""
Model training script for the restaurant recommendation system
Trains content-based filtering models using the cleaned dataset
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_combined_features(df):
    """
    Create combined text features for content-based filtering
    
    Args:
        df: Cleaned dataframe
        
    Returns:
        Series of combined features
    """
    logger.info("Creating combined features...")
    
    # Combine relevant text features
    features = []
    for _, row in df.iterrows():
        feature_parts = []
        
        # Add restaurant name
        if pd.notna(row['Restaurant_Name']):
            feature_parts.append(str(row['Restaurant_Name']))
        
        # Add cuisines
        if pd.notna(row['Cuisines']):
            feature_parts.append(str(row['Cuisines']))
        
        # Add restaurant type
        if pd.notna(row['Restaurant_Type']):
            feature_parts.append(str(row['Restaurant_Type']))
        
        # Add location
        if pd.notna(row['Location']):
            feature_parts.append(str(row['Location']))
        
        # Add city
        if pd.notna(row['City']):
            feature_parts.append(str(row['City']))
        
        # Add dish liked
        if pd.notna(row['Dish_Liked']) and str(row['Dish_Liked']).strip():
            feature_parts.append(str(row['Dish_Liked']))
        
        # Add price category
        if pd.notna(row['Price_Category']):
            feature_parts.append(str(row['Price_Category']))
        
        # Add rating category
        if pd.notna(row['Rating_Category']):
            feature_parts.append(str(row['Rating_Category']))
        
        # Join all features
        combined = ' '.join(feature_parts)
        features.append(combined)
    
    return pd.Series(features)

def train_recommendation_models(data_file: str, models_dir: str):
    """
    Train recommendation models using the cleaned dataset
    
    Args:
        data_file: Path to the cleaned dataset
        models_dir: Directory to save the trained models
    """
    logger.info(f"Loading cleaned dataset from {data_file}")
    
    # Load the cleaned dataset
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} restaurants")
    
    # Create combined features
    df['Combined_Features'] = create_combined_features(df)
    
    # Initialize TF-IDF vectorizer
    logger.info("Initializing TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    # Fit and transform the combined features
    logger.info("Fitting TF-IDF vectorizer...")
    tfidf_matrix = tfidf.fit_transform(df['Combined_Features'])
    
    # Calculate cosine similarity matrix
    logger.info("Calculating cosine similarity matrix...")
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Create a scaler for numerical features
    logger.info("Creating feature scaler...")
    numerical_features = ['Rating', 'Votes', 'Price']
    scaler = StandardScaler()
    
    # Prepare numerical features for scaling
    numerical_data = df[numerical_features].fillna(0)
    scaler.fit(numerical_data)
    
    # Create model data structure
    model_data = {
        'tfidf_vectorizer': tfidf,
        'similarity_matrix': similarity_matrix,
        'scaler': scaler,
        'restaurant_data': df[['Restaurant_Name', 'Cuisines', 'Location', 'City', 
                              'Rating', 'Votes', 'Price', 'Restaurant_Type', 
                              'Price_Category', 'Rating_Category']].to_dict('records'),
        'feature_names': tfidf.get_feature_names_out().tolist(),
        'model_info': {
            'total_restaurants': len(df),
            'feature_count': len(tfidf.get_feature_names_out()),
            'similarity_shape': similarity_matrix.shape
        }
    }
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the main model
    model_path = os.path.join(models_dir, 'model.pkl')
    logger.info(f"Saving main model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    # Save similarity matrix separately for faster loading
    similarity_path = os.path.join(models_dir, 'similarity.pkl')
    logger.info(f"Saving similarity matrix to {similarity_path}")
    with open(similarity_path, 'wb') as f:
        pickle.dump(similarity_matrix, f)
    
    # Test the models
    logger.info("Testing trained models...")
    
    # Test content-based recommendation
    test_query = "pizza italian"
    query_vector = tfidf.transform([test_query])
    query_similarity = cosine_similarity(query_vector, tfidf_matrix)[0]
    
    # Get top 5 similar restaurants
    top_indices = query_similarity.argsort()[-5:][::-1]
    top_restaurants = [df.iloc[idx]['Restaurant_Name'] for idx in top_indices if query_similarity[idx] > 0.01]
    
    logger.info(f"Test query '{test_query}' found {len(top_restaurants)} similar restaurants:")
    for i, restaurant in enumerate(top_restaurants[:3], 1):
        logger.info(f"  {i}. {restaurant}")
    
    # Model statistics
    logger.info("\n" + "="*50)
    logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*50)
    logger.info(f"Total restaurants: {len(df)}")
    logger.info(f"Feature dimensions: {tfidf_matrix.shape[1]}")
    logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
    logger.info(f"Models saved to: {models_dir}")
    logger.info(f"  - model.pkl: Main model with vectorizer and data")
    logger.info(f"  - similarity.pkl: Precomputed similarity matrix")
    
    return model_data

def load_models(models_dir: str):
    """
    Load trained models for testing
    
    Args:
        models_dir: Directory containing the trained models
        
    Returns:
        Loaded model data
    """
    model_path = os.path.join(models_dir, 'model.pkl')
    similarity_path = os.path.join(models_dir, 'similarity.pkl')
    
    logger.info(f"Loading models from {models_dir}")
    
    # Load main model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Load similarity matrix
    with open(similarity_path, 'rb') as f:
        similarity_matrix = pickle.load(f)
    
    # Verify loaded data
    logger.info(f"Loaded model with {len(model_data['restaurant_data'])} restaurants")
    logger.info(f"Feature count: {len(model_data['feature_names'])}")
    logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    return model_data, similarity_matrix

if __name__ == "__main__":
    # Paths
    data_file = "data/zomato_cleaned.csv"
    models_dir = "models"
    
    try:
        # Train the models
        model_data = train_recommendation_models(data_file, models_dir)
        
        # Test loading the models
        logger.info("\nTesting model loading...")
        loaded_model, loaded_similarity = load_models(models_dir)
        
        logger.info("✅ All tests passed! Models are ready for use.")
        
    except Exception as e:
        logger.error(f"❌ Error during model training: {e}")
        raise
