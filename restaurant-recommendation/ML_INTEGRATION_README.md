# 🤖 ML-Powered Restaurant Recommendation System

This document describes the advanced machine learning integration that has been added to the Restaurant Recommendation Flask application.

## 🎯 Overview

The system now includes a comprehensive ML recommendation engine that combines:

1. **Content-Based Filtering** - Uses TF-IDF and cosine similarity
2. **Collaborative Filtering** - Uses LightFM for user-item interactions
3. **Context-Aware Features** - Uses SentenceTransformers for mood/time/occasion matching
4. **Hybrid Recommendations** - Combines all models for optimal results

## 🚀 Features

### Advanced ML Models
- **TF-IDF Vectorization** for text-based restaurant matching
- **LightFM** for collaborative filtering and user preferences
- **SentenceTransformers** for semantic understanding of moods and contexts
- **Cosine Similarity** for content-based recommendations
- **Hybrid Scoring** that combines all models with weighted averages

### Context-Aware Recommendations
- **Mood-based filtering**: Happy, Sad, Angry, Relaxed, Excited, Bored
- **Time-based filtering**: Morning, Afternoon, Evening, Night
- **Occasion-based filtering**: Birthday, Date, Party, Lunch, Dinner, Meeting, Anniversary

### Production-Ready Features
- **Model Persistence** - Saves trained models as `restaurant_predictor.joblib`
- **Flask Caching** - 5-minute cache for faster responses
- **Error Handling** - Graceful fallbacks when models aren't available
- **Background Training** - Non-blocking model initialization
- **RESTful API** - Clean JSON API for recommendations

## 📁 File Structure

```
restaurant-recommendation/
├── ml_recommendation_engine.py    # Main ML engine
├── train_ml_models.py            # Training script
├── test_ml_engine.py             # Testing script
├── app.py                        # Updated Flask app with ML integration
├── requirements.txt              # Updated with ML dependencies
├── templates/home.html           # Updated with AI recommendation UI
└── restaurant_predictor.joblib   # Trained models (generated)
```

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
cd restaurant-recommendation
pip install -r requirements.txt
```

### 2. Train the ML Models

```bash
python train_ml_models.py
```

This will:
- Load and preprocess the `zomato.csv` dataset
- Train all ML models (TF-IDF, LightFM, SentenceTransformers)
- Save the trained models to `restaurant_predictor.joblib`

### 3. Start the Flask Application

```bash
python app.py
```

The ML engine will initialize automatically in the background.

## 🔧 API Endpoints

### Get AI Recommendations

**POST** `/recommend`

```json
{
  "user_input": "North Indian food",
  "mood": "happy",
  "time": "evening",
  "occasion": "dinner"
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "name": "Empire Restaurant",
      "cuisines": "North Indian, Mughlai",
      "rating": 4.5,
      "address": "Church Street, Bangalore",
      "cost": 800,
      "match_score": 0.97,
      "type": "hybrid"
    }
  ],
  "query": {
    "user_input": "North Indian food",
    "mood": "happy",
    "time": "evening",
    "occasion": "dinner"
  },
  "total": 5
}
```

### Check ML Engine Status

**GET** `/ml/status`

**Response:**
```json
{
  "available": true,
  "models_loaded": true,
  "restaurant_count": 51717,
  "content_based": true,
  "collaborative": true,
  "context_aware": true
}
```

## 🎨 Frontend Integration

The home page now includes an AI-powered recommendation interface:

- **Food Preference Input** - What kind of food you're craving
- **Mood Selector** - How you're feeling (Happy, Sad, etc.)
- **Time Selector** - What time it is (Morning, Evening, etc.)
- **Occasion Selector** - What's the occasion (Date, Birthday, etc.)
- **Real-time Results** - Displays recommendations with match scores

## 🧪 Testing

### Test the ML Engine Directly

```bash
python test_ml_engine.py
```

### Test via API (with Flask running)

```bash
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "pizza",
    "mood": "happy",
    "time": "evening",
    "occasion": "dinner"
  }'
```

## 🔍 How It Works

### 1. Data Preprocessing
- Loads `zomato.csv` dataset
- Cleans text data (reviews, cuisines, names)
- Combines features for ML training
- Handles missing values and encoding

### 2. Content-Based Filtering
- Uses TF-IDF vectorization on combined restaurant features
- Calculates cosine similarity between user queries and restaurants
- Recommends similar restaurants based on content

### 3. Collaborative Filtering
- Simulates user interactions based on ratings and votes
- Uses LightFM to learn user preferences
- Provides personalized recommendations

### 4. Context-Aware Features
- Maps moods, times, and occasions to semantic embeddings
- Uses SentenceTransformers for semantic understanding
- Matches context with restaurant characteristics

### 5. Hybrid Recommendations
- Combines all three models with weighted scores:
  - Content-based: 40% weight
  - Collaborative: 30% weight
  - Context-aware: 30% weight
- Adds diversity boost for restaurants appearing in multiple models
- Returns top N recommendations sorted by final score

## ⚡ Performance Features

- **Caching**: 5-minute cache for recommendation requests
- **Background Loading**: Models load in background threads
- **Error Handling**: Graceful fallbacks when models aren't ready
- **Memory Efficient**: Only loads necessary model components
- **Fast Inference**: Optimized for real-time recommendations

## 🎯 Model Accuracy

The hybrid approach provides:
- **High Relevance**: Content-based filtering ensures food matches
- **Personalization**: Collaborative filtering learns user preferences
- **Context Awareness**: Mood/time/occasion matching for better UX
- **Diversity**: Multiple models prevent overfitting to one approach

## 🔧 Configuration

### Model Parameters

In `ml_recommendation_engine.py`:

```python
# TF-IDF Parameters
max_features=5000
ngram_range=(1, 3)
min_df=2
max_df=0.95

# LightFM Parameters
no_components=50
learning_rate=0.05
loss='warp'

# Hybrid Weights
content_weight = 0.4
collaborative_weight = 0.3
context_weight = 0.3
```

### Cache Settings

In `app.py`:

```python
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300  # 5 minutes
})
```

## 🚨 Troubleshooting

### Common Issues

1. **Models not loading**: Ensure `zomato.csv` exists and run `train_ml_models.py`
2. **Memory issues**: Reduce `max_features` in TF-IDF or use smaller datasets
3. **Slow responses**: Check if models are loaded via `/ml/status`
4. **Import errors**: Install all dependencies with `pip install -r requirements.txt`

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Dataset Requirements

The system expects `zomato.csv` with these columns:
- `name`: Restaurant name
- `cuisines`: Cuisine types
- `location`: Restaurant location
- `rate`: Rating (0-5)
- `votes`: Number of votes
- `approx_cost(for two people)`: Cost for two people
- `reviews_list`: Customer reviews
- `rest_type`: Restaurant type
- `dish_liked`: Popular dishes

## 🔮 Future Enhancements

- **Real-time Learning**: Update models based on user interactions
- **A/B Testing**: Compare different recommendation strategies
- **Deep Learning**: Add neural networks for better semantic understanding
- **Multi-language**: Support for multiple languages
- **Geographic Filtering**: Location-based recommendations
- **Dietary Restrictions**: Filter by dietary preferences

## 📝 License

This ML integration is part of the Restaurant Recommendation System and follows the same license terms.

---

**🎉 The ML-powered recommendation system is now ready to provide intelligent, context-aware restaurant suggestions!**
