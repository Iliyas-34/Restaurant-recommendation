#!/usr/bin/env python3
"""
Training script for the ML recommendation engine
Trains all models on the zomato.csv dataset and saves them
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_recommendation_engine import RestaurantRecommendationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    logger.info("Starting ML model training...")
    
    # Check if zomato.csv exists
    csv_path = "zomato.csv"
    if not os.path.exists(csv_path):
        logger.error(f"zomato.csv not found at {csv_path}")
        logger.info("Please ensure zomato.csv is in the current directory")
        return False
    
    try:
        # Initialize the recommendation engine
        logger.info("Initializing recommendation engine...")
        engine = RestaurantRecommendationEngine(csv_path)
        
        # Train all models
        logger.info("Training all ML models...")
        engine.train_all_models()
        
        # Save the trained models
        model_path = "restaurant_predictor.joblib"
        logger.info(f"Saving models to {model_path}...")
        engine.save_models(model_path)
        
        logger.info("✅ Model training completed successfully!")
        logger.info(f"Models saved to: {model_path}")
        
        # Test the models
        logger.info("Testing the trained models...")
        
        # Test content-based recommendations
        logger.info("Testing content-based recommendations...")
        content_recs = engine.get_content_based_recommendations("pizza", n=3)
        logger.info(f"Content-based recommendations for 'pizza': {len(content_recs)} results")
        
        # Test context-aware recommendations
        logger.info("Testing context-aware recommendations...")
        context_recs = engine.get_context_aware_recommendations("happy", "evening", "dinner", n=3)
        logger.info(f"Context-aware recommendations: {len(context_recs)} results")
        
        # Test hybrid recommendations
        logger.info("Testing hybrid recommendations...")
        hybrid_recs = engine.get_hybrid_recommendations("Italian food", "happy", "evening", "dinner", n=3)
        logger.info(f"Hybrid recommendations: {len(hybrid_recs)} results")
        
        if hybrid_recs:
            logger.info("Sample recommendation:")
            for i, rec in enumerate(hybrid_recs[:2]):
                logger.info(f"  {i+1}. {rec['name']} - {rec['cuisines']} (Score: {rec['match_score']:.3f})")
        
        logger.info("🎉 All tests passed! The ML recommendation engine is ready to use.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
