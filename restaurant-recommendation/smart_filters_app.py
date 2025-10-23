#!/usr/bin/env python3
"""
Simplified Flask app for Restaurant Smart Filters
This app focuses on the core functionality: smart filters with real data
"""

from flask import Flask, render_template, jsonify, request
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import restaurant processor
from utils.restaurant_processor import RestaurantDataProcessor

app = Flask(__name__)

# Global processor instance
processor = None

def initialize_processor():
    global processor
    print("Initializing Restaurant Data Processor...")
    processor = RestaurantDataProcessor()
    success = processor.load_restaurants()
    if success:
        print(f"✅ Successfully loaded {len(processor.restaurants)} restaurants")
        return True
    else:
        print("❌ Failed to load restaurants")
        return False

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/api/smart-filters", methods=["POST"])
def smart_filters_recommendations():
    """
    Smart filters endpoint that takes mood, time, and occasion filters
    and returns contextual restaurant recommendations from restaurants.json
    """
    try:
        print("=== SMART FILTERS ENDPOINT CALLED ===")
        
        data = request.get_json()
        print(f"Received data: {data}")
        
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "recommendations": []
            }), 400
        
        # Extract filter parameters
        mood = data.get("mood", "").strip().lower()
        time = data.get("time", "").strip().lower()
        occasion = data.get("occasion", "").strip().lower()
        
        print(f"Processing filters: mood={mood}, time={time}, occasion={occasion}")
        
        if not processor:
            return jsonify({
                "error": "Restaurant processor not initialized",
                "recommendations": []
            }), 500
        
        # Get recommendations from JSON data
        recommendations = processor.get_smart_recommendations(mood, time, occasion, limit=12)
        
        if recommendations:
            # Format recommendations for frontend
            formatted_recommendations = []
            for rec in recommendations:
                formatted_recommendations.append({
                    "name": rec.get("name", ""),
                    "cuisines": rec.get("cuisines", ""),
                    "rating": rec.get("rating", 0),
                    "address": rec.get("address", ""),
                    "cost": rec.get("cost", 0),
                    "match_score": rec.get("match_score", 0),
                    "restaurant_type": rec.get("restaurant_type", ""),
                    "dish_liked": rec.get("dish_liked", ""),
                    "online_order": rec.get("online_order", False),
                    "book_table": rec.get("book_table", False),
                    "price_category": rec.get("price_category", "moderate")
                })
            
            print(f"Returning {len(formatted_recommendations)} recommendations from JSON data")
            
            return jsonify({
                "recommendations": formatted_recommendations,
                "filters": {
                    "mood": mood,
                    "time": time,
                    "occasion": occasion,
                    "search_query": ""
                },
                "total": len(formatted_recommendations),
                "engine_used": "json_data_processor"
            })
        else:
            print("No recommendations found, trying fallback")
            fallback_recs = processor.get_fallback_recommendations(limit=12)
            if fallback_recs:
                formatted_fallback = []
                for rec in fallback_recs:
                    formatted_fallback.append({
                        "name": rec.get("name", ""),
                        "cuisines": rec.get("cuisines", ""),
                        "rating": rec.get("rating", 0),
                        "address": rec.get("address", ""),
                        "cost": rec.get("cost", 0),
                        "match_score": 0.8,
                        "restaurant_type": rec.get("restaurant_type", ""),
                        "dish_liked": rec.get("dish_liked", ""),
                        "online_order": rec.get("online_order", False),
                        "book_table": rec.get("book_table", False),
                        "price_category": rec.get("price_category", "moderate")
                    })
                
                return jsonify({
                    "recommendations": formatted_fallback,
                    "filters": {"mood": mood, "time": time, "occasion": occasion},
                    "total": len(formatted_fallback),
                    "engine_used": "fallback_json"
                })
            else:
                return jsonify({
                    "error": "No restaurants found",
                    "recommendations": []
                }), 404
        
    except Exception as e:
        print(f"Error in smart filters: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": "Unable to process request",
            "recommendations": []
        }), 500

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Fallback recommendation endpoint
    """
    try:
        data = request.get_json()
        user_input = data.get("user_input", "").strip()
        
        if not processor:
            return jsonify({"recommendations": []})
        
        # Get fallback recommendations
        recommendations = processor.get_fallback_recommendations(limit=10)
        
        formatted_recommendations = []
        for rec in recommendations:
            formatted_recommendations.append({
                "name": rec.get("name", ""),
                "cuisines": rec.get("cuisines", ""),
                "rating": rec.get("rating", 0),
                "location": rec.get("address", ""),
                "cost": rec.get("cost", 0),
                "restaurant_type": rec.get("restaurant_type", ""),
                "dish_liked": rec.get("dish_liked", ""),
                "online_order": rec.get("online_order", False),
                "book_table": rec.get("book_table", False)
            })
        
        return jsonify({"recommendations": formatted_recommendations})
        
    except Exception as e:
        print(f"Error in recommend endpoint: {e}")
        return jsonify({"recommendations": []})

if __name__ == "__main__":
    print("🚀 Starting Restaurant Smart Filters App...")
    print("📍 Server will be available at: http://localhost:5000")
    print("🎯 Smart filters endpoint: POST http://localhost:5000/api/smart-filters")
    
    # Initialize processor
    if initialize_processor():
        print("✅ Ready to serve recommendations!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize. Exiting.")
        sys.exit(1)
