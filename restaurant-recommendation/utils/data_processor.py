#!/usr/bin/env python3
"""
Data Processing and Normalization for Restaurant Recommendation System
Handles cleaning, normalization, and geocoding of Zomato dataset
"""

import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Tuple, Optional
import logging
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RestaurantDataProcessor:
    """Comprehensive data processor for restaurant recommendation system"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.geocoder = Nominatim(user_agent="restaurant_finder")
        
    def load_data(self) -> pd.DataFrame:
        """Load and perform initial data cleaning"""
        try:
            logger.info(f"Loading data from {self.csv_path}")
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.df)} restaurants")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_restaurant_names(self) -> None:
        """Clean and standardize restaurant names"""
        logger.info("Cleaning restaurant names...")
        self.df['Restaurant_Name'] = self.df['Restaurant_Name'].str.strip()
        self.df['Restaurant_Name'] = self.df['Restaurant_Name'].str.replace(r'\s+', ' ', regex=True)
        
        # Remove special characters but keep essential ones
        self.df['Restaurant_Name'] = self.df['Restaurant_Name'].str.replace(r'[^\w\s\-&\.\']', '', regex=True)
        
        # Fill missing names
        missing_names = self.df['Restaurant_Name'].isna().sum()
        if missing_names > 0:
            logger.warning(f"Found {missing_names} missing restaurant names, filling with 'Unknown Restaurant'")
            self.df['Restaurant_Name'] = self.df['Restaurant_Name'].fillna('Unknown Restaurant')
    
    def normalize_cuisines(self) -> None:
        """Normalize and clean cuisine data"""
        logger.info("Normalizing cuisines...")
        
        # Clean cuisine strings
        self.df['Cuisines'] = self.df['Cuisines'].fillna('Unknown')
        self.df['Cuisines'] = self.df['Cuisines'].astype(str)
        
        # Standardize common cuisine names
        cuisine_mapping = {
            'North Indian': ['North Indian', 'Indian', 'North India'],
            'Chinese': ['Chinese', 'Chineese', 'Chinise'],
            'Italian': ['Italian', 'Italain', 'Itallian'],
            'American': ['American', 'Americain', 'Amercian'],
            'Mexican': ['Mexican', 'Mexicain', 'Mexican'],
            'Thai': ['Thai', 'Thailand', 'Thia'],
            'Japanese': ['Japanese', 'Japan', 'Japaneese'],
            'Korean': ['Korean', 'Korea', 'Koreaan'],
            'Mediterranean': ['Mediterranean', 'Mediteranean', 'Mediterranian'],
            'Continental': ['Continental', 'Continentel', 'Continentail'],
            'Fast Food': ['Fast Food', 'Fastfood', 'Fast-Food'],
            'Street Food': ['Street Food', 'Streetfood', 'Street-Food'],
            'Desserts': ['Desserts', 'Dessert', 'Deserts'],
            'Beverages': ['Beverages', 'Beverage', 'Drinks'],
            'Bakery': ['Bakery', 'Bakary', 'Bakry'],
            'Cafe': ['Cafe', 'Coffee', 'Café', 'Coffe'],
            'BBQ': ['BBQ', 'Barbeque', 'Barbecue', 'Bar-B-Q'],
            'Seafood': ['Seafood', 'Sea Food', 'Sea-Food'],
            'Vegan': ['Vegan', 'Vegetarian', 'Veg'],
            'Fusion': ['Fusion', 'Fusian', 'Fuson']
        }
        
        def normalize_cuisine_list(cuisine_str):
            if pd.isna(cuisine_str) or cuisine_str == 'Unknown':
                return 'Unknown'
            
            cuisines = [c.strip() for c in str(cuisine_str).split(',')]
            normalized = []
            
            for cuisine in cuisines:
                cuisine = cuisine.strip().title()
                # Check if cuisine matches any known mapping
                for standard, variants in cuisine_mapping.items():
                    if any(variant.lower() == cuisine.lower() for variant in variants):
                        normalized.append(standard)
                        break
                else:
                    # If no mapping found, use the original (cleaned)
                    normalized.append(cuisine)
            
            return ', '.join(sorted(set(normalized)))
        
        self.df['Cuisines'] = self.df['Cuisines'].apply(normalize_cuisine_list)
    
    def clean_ratings(self) -> None:
        """Clean and normalize rating data"""
        logger.info("Cleaning ratings...")
        
        # Convert to numeric, handling various formats
        self.df['Rating'] = pd.to_numeric(self.df['Rating'], errors='coerce')
        
        # Fill missing ratings with median
        median_rating = self.df['Rating'].median()
        self.df['Rating'] = self.df['Rating'].fillna(median_rating)
        
        # Ensure ratings are between 0 and 5
        self.df['Rating'] = self.df['Rating'].clip(0, 5)
        
        # Create rating categories
        def categorize_rating(rating):
            if rating >= 4.5:
                return 'Excellent'
            elif rating >= 4.0:
                return 'Very Good'
            elif rating >= 3.5:
                return 'Good'
            elif rating >= 3.0:
                return 'Average'
            else:
                return 'Poor'
        
        self.df['Rating_Category'] = self.df['Rating'].apply(categorize_rating)
    
    def clean_prices(self) -> None:
        """Clean and normalize price data"""
        logger.info("Cleaning prices...")
        
        # Convert to numeric
        self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce')
        
        # Fill missing prices with median
        median_price = self.df['Price'].median()
        self.df['Price'] = self.df['Price'].fillna(median_price)
        
        # Create price categories
        def categorize_price(price):
            if price <= 300:
                return 'Budget'
            elif price <= 600:
                return 'Moderate'
            elif price <= 1200:
                return 'Expensive'
            else:
                return 'Very Expensive'
        
        self.df['Price_Category'] = self.df['Price'].apply(categorize_price)
        
        # Create price range strings
        self.df['Price_Range'] = self.df['Price'].apply(
            lambda x: f"₹{int(x)}" if not pd.isna(x) else "₹0"
        )
    
    def clean_locations(self) -> None:
        """Clean and normalize location data"""
        logger.info("Cleaning locations...")
        
        # Clean city names
        self.df['City'] = self.df['City'].fillna('Unknown City')
        self.df['City'] = self.df['City'].str.strip().str.title()
        
        # Clean locality
        self.df['Location'] = self.df['Location'].fillna('Unknown Area')
        self.df['Location'] = self.df['Location'].str.strip()
        
        # Clean address
        self.df['Address'] = self.df['Address'].fillna('Address not available')
        self.df['Address'] = self.df['Address'].str.strip()
        
        # Create full location string
        self.df['Full_Location'] = self.df['Location'] + ', ' + self.df['City']
    
    def geocode_locations(self, sample_size: int = 100) -> None:
        """Geocode locations to get latitude/longitude coordinates"""
        logger.info(f"Geocoding locations (sample of {sample_size})...")
        
        # Initialize coordinate columns
        self.df['Latitude'] = np.nan
        self.df['Longitude'] = np.nan
        
        # Sample for geocoding to avoid rate limits
        sample_df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)
        
        for idx, row in sample_df.iterrows():
            try:
                location = f"{row['Location']}, {row['City']}, India"
                geocode_result = self.geocoder.geocode(location, timeout=10)
                
                if geocode_result:
                    self.df.at[idx, 'Latitude'] = geocode_result.latitude
                    self.df.at[idx, 'Longitude'] = geocode_result.longitude
                    logger.info(f"Geocoded: {location} -> {geocode_result.latitude}, {geocode_result.longitude}")
                
                # Rate limiting
                time.sleep(1)
                
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                logger.warning(f"Geocoding failed for {location}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error geocoding {location}: {e}")
                continue
        
        # Fill missing coordinates with city center approximations
        city_coords = {
            'Bangalore': (12.9716, 77.5946),
            'Delhi': (28.7041, 77.1025),
            'Mumbai': (19.0760, 72.8777),
            'Kolkata': (22.5726, 88.3639),
            'Chennai': (13.0827, 80.2707),
            'Hyderabad': (17.3850, 78.4867),
            'Pune': (18.5204, 73.8567),
            'Ahmedabad': (23.0225, 72.5714),
            'Jaipur': (26.9124, 75.7873),
            'Surat': (21.1702, 72.8311)
        }
        
        for city, (lat, lng) in city_coords.items():
            mask = (self.df['City'].str.contains(city, case=False, na=False)) & \
                   (self.df['Latitude'].isna())
            self.df.loc[mask, 'Latitude'] = lat
            self.df.loc[mask, 'Longitude'] = lng
    
    def clean_reviews(self) -> None:
        """Clean and process review data"""
        logger.info("Cleaning reviews...")
        
        # Clean votes
        self.df['Votes'] = pd.to_numeric(self.df['Votes'], errors='coerce')
        self.df['Votes'] = self.df['Votes'].fillna(0)
        
        # Process review text
        if 'Reviews_List' in self.df.columns:
            self.df['Reviews_List'] = self.df['Reviews_List'].fillna('[]')
            
            def clean_review_text(review_str):
                if pd.isna(review_str) or review_str == '[]':
                    return []
                
                try:
                    # Parse the review string
                    reviews = eval(review_str) if isinstance(review_str, str) else review_str
                    if isinstance(reviews, list):
                        return [str(review).strip() for review in reviews if review]
                    return []
                except:
                    return []
            
            self.df['Reviews_List'] = self.df['Reviews_List'].apply(clean_review_text)
    
    def create_combined_features(self) -> None:
        """Create combined features for ML processing"""
        logger.info("Creating combined features...")
        
        # Combine text features
        text_columns = ['Restaurant_Name', 'Cuisines', 'Location', 'City']
        
        def combine_text_features(row):
            features = []
            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    features.append(str(row[col]))
            return ' '.join(features)
        
        self.df['Combined_Features'] = self.df.apply(combine_text_features, axis=1)
        
        # Create cuisine list for easier processing
        self.df['Cuisine_List'] = self.df['Cuisines'].apply(
            lambda x: [c.strip() for c in str(x).split(',')] if pd.notna(x) else []
        )
    
    def remove_duplicates(self) -> None:
        """Remove duplicate restaurants"""
        logger.info("Removing duplicates...")
        
        initial_count = len(self.df)
        
        # Remove exact duplicates
        self.df = self.df.drop_duplicates(subset=['Restaurant_Name', 'Location', 'City'])
        
        # Remove near-duplicates based on name similarity
        self.df = self.df.drop_duplicates(subset=['Restaurant_Name'], keep='first')
        
        final_count = len(self.df)
        removed = initial_count - final_count
        logger.info(f"Removed {removed} duplicate restaurants")
    
    def filter_quality_data(self) -> None:
        """Filter out low-quality data"""
        logger.info("Filtering quality data...")
        
        initial_count = len(self.df)
        
        # Remove restaurants with missing essential data
        essential_columns = ['Restaurant_Name', 'Cuisines', 'City']
        self.df = self.df.dropna(subset=essential_columns)
        
        # Remove restaurants with very low ratings (likely data errors)
        self.df = self.df[self.df['Rating'] >= 1.0]
        
        # Remove restaurants with unrealistic prices
        self.df = self.df[(self.df['Price'] >= 50) & 
                         (self.df['Price'] <= 10000)]
        
        final_count = len(self.df)
        removed = initial_count - final_count
        logger.info(f"Filtered out {removed} low-quality entries")
    
    def process_all(self) -> pd.DataFrame:
        """Run complete data processing pipeline"""
        logger.info("Starting complete data processing pipeline...")
        
        # Load data
        self.load_data()
        
        # Clean and normalize
        self.clean_restaurant_names()
        self.normalize_cuisines()
        self.clean_ratings()
        self.clean_prices()
        self.clean_locations()
        self.clean_reviews()
        
        # Create features
        self.create_combined_features()
        
        # Quality control
        self.remove_duplicates()
        self.filter_quality_data()
        
        # Geocode (optional, can be time-consuming)
        try:
            self.geocode_locations(sample_size=50)
        except Exception as e:
            logger.warning(f"Geocoding failed: {e}")
        
        logger.info(f"Data processing complete. Final dataset: {len(self.df)} restaurants")
        return self.df
    
    def save_processed_data(self, output_path: str) -> None:
        """Save processed data to CSV and JSON"""
        logger.info(f"Saving processed data to {output_path}")
        
        # Save as CSV
        csv_path = output_path.replace('.json', '.csv')
        self.df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")
        
        # Save as JSON for web app
        json_data = self.df.to_dict(orient='records')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON: {output_path}")
        
        # Save summary statistics
        stats = {
            'total_restaurants': len(self.df),
            'cities': self.df['City'].nunique(),
            'cuisines': len(set([c for cuisine_list in self.df['Cuisine_List'] for c in cuisine_list])),
            'avg_rating': self.df['Rating'].mean(),
            'avg_price': self.df['Price'].mean(),
            'rating_distribution': self.df['Rating_Category'].value_counts().to_dict(),
            'price_distribution': self.df['Price_Category'].value_counts().to_dict()
        }
        
        stats_path = output_path.replace('.json', '_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved statistics: {stats_path}")

def main():
    """Main processing function"""
    processor = RestaurantDataProcessor('data/zomato_cleaned.csv')
    
    try:
        # Process data
        processed_df = processor.process_all()
        
        # Save results
        processor.save_processed_data('data/restaurants_processed.json')
        
        print("[OK] Data processing completed successfully!")
        print(f"[INFO] Processed {len(processed_df)} restaurants")
        print(f"[INFO] {processed_df['City'].nunique()} cities")
        print(f"[INFO] {len(set([c for cuisine_list in processed_df['Cuisine_List'] for c in cuisine_list]))} unique cuisines")
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
