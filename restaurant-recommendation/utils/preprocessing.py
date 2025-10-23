#!/usr/bin/env python3
"""
Data preprocessing utilities for the restaurant recommendation system
Cleans and standardizes the zomato.csv dataset
"""

import pandas as pd
import numpy as np
import re
import os
from typing import Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_restaurant_data(input_file: str, output_file: str) -> Tuple[pd.DataFrame, dict]:
    """
    Clean and standardize the restaurant dataset
    
    Args:
        input_file: Path to the original zomato.csv file
        output_file: Path to save the cleaned dataset
        
    Returns:
        Tuple of (cleaned_dataframe, cleaning_stats)
    """
    logger.info(f"Loading dataset from {input_file}")
    
    # Load the dataset
    df = pd.read_csv(input_file)
    original_shape = df.shape
    logger.info(f"Original dataset shape: {original_shape}")
    
    # Initialize cleaning stats
    stats = {
        'original_rows': original_shape[0],
        'original_columns': original_shape[1],
        'rows_removed': 0,
        'columns_removed': 0,
        'missing_values_filled': 0
    }
    
    # 1. Remove unnecessary columns
    columns_to_remove = [
        'url', 'phone', 'menu_item', 'listed_in(type)', 'listed_in(city)'
    ]
    
    existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    df = df.drop(columns=existing_columns_to_remove)
    stats['columns_removed'] = len(existing_columns_to_remove)
    logger.info(f"Removed {len(existing_columns_to_remove)} unnecessary columns")
    
    # 2. Standardize column names
    column_mapping = {
        'name': 'Restaurant_Name',
        'cuisines': 'Cuisines',
        'location': 'Location',
        'rate': 'Rating',
        'approx_cost(for two people)': 'Price',
        'votes': 'Votes',
        'rest_type': 'Restaurant_Type',
        'dish_liked': 'Dish_Liked',
        'online_order': 'Online_Order',
        'book_table': 'Book_Table',
        'address': 'Address',
        'reviews_list': 'Reviews_List'
    }
    
    df = df.rename(columns=column_mapping)
    logger.info("Standardized column names")
    
    # 3. Remove rows with missing essential data
    essential_columns = ['Restaurant_Name', 'Cuisines', 'Location', 'Rating']
    
    # Check for missing values in essential columns
    missing_essential = df[essential_columns].isnull().any(axis=1)
    rows_before = len(df)
    df = df[~missing_essential]
    rows_after = len(df)
    stats['rows_removed'] += (rows_before - rows_after)
    logger.info(f"Removed {rows_before - rows_after} rows with missing essential data")
    
    # 4. Clean and standardize text data
    def clean_text(text):
        if pd.isna(text):
            return ""
        return str(text).strip()
    
    # Clean restaurant names
    df['Restaurant_Name'] = df['Restaurant_Name'].apply(clean_text)
    
    # Clean cuisines
    df['Cuisines'] = df['Cuisines'].apply(clean_text)
    
    # Clean locations
    df['Location'] = df['Location'].apply(clean_text)
    
    # Clean restaurant types
    df['Restaurant_Type'] = df['Restaurant_Type'].apply(clean_text)
    
    # Clean dish liked
    df['Dish_Liked'] = df['Dish_Liked'].apply(clean_text)
    
    # Clean addresses
    df['Address'] = df['Address'].apply(clean_text)
    
    logger.info("Cleaned text data")
    
    # 5. Clean and standardize ratings
    def clean_rating(rating):
        if pd.isna(rating):
            return 0.0
        
        # Convert to string and clean
        rating_str = str(rating).strip()
        
        # Remove '/5' suffix if present
        if '/5' in rating_str:
            rating_str = rating_str.replace('/5', '')
        
        # Extract numeric value
        try:
            return float(rating_str)
        except (ValueError, TypeError):
            return 0.0
    
    df['Rating'] = df['Rating'].apply(clean_rating)
    
    # Remove restaurants with rating 0 (invalid ratings)
    rows_before = len(df)
    df = df[df['Rating'] > 0]
    rows_after = len(df)
    stats['rows_removed'] += (rows_before - rows_after)
    logger.info(f"Removed {rows_before - rows_after} rows with invalid ratings")
    
    # 6. Clean and standardize prices
    def clean_price(price):
        if pd.isna(price):
            return 0
        
        # Convert to string and clean
        price_str = str(price).strip()
        
        # Remove currency symbols and commas
        price_str = re.sub(r'[₹,$,\s]', '', price_str)
        
        try:
            return int(float(price_str))
        except (ValueError, TypeError):
            return 0
    
    df['Price'] = df['Price'].apply(clean_price)
    
    # 7. Clean votes
    def clean_votes(votes):
        if pd.isna(votes):
            return 0
        
        try:
            return int(float(str(votes).strip()))
        except (ValueError, TypeError):
            return 0
    
    df['Votes'] = df['Votes'].apply(clean_votes)
    
    # 8. Clean boolean columns
    def clean_boolean(value):
        if pd.isna(value):
            return False
        
        value_str = str(value).strip().lower()
        return value_str in ['yes', 'true', '1', 'y']
    
    df['Online_Order'] = df['Online_Order'].apply(clean_boolean)
    df['Book_Table'] = df['Book_Table'].apply(clean_boolean)
    
    # 9. Clean reviews list
    def clean_reviews(reviews):
        if pd.isna(reviews):
            return ""
        
        reviews_str = str(reviews).strip()
        
        # Remove extra whitespace and normalize
        reviews_str = re.sub(r'\s+', ' ', reviews_str)
        
        return reviews_str
    
    df['Reviews_List'] = df['Reviews_List'].apply(clean_reviews)
    
    # 10. Remove duplicate restaurants (based on name and location)
    rows_before = len(df)
    df = df.drop_duplicates(subset=['Restaurant_Name', 'Location'], keep='first')
    rows_after = len(df)
    stats['rows_removed'] += (rows_before - rows_after)
    logger.info(f"Removed {rows_before - rows_after} duplicate restaurants")
    
    # 11. Fill remaining missing values
    missing_before = df.isnull().sum().sum()
    
    # Fill missing values with appropriate defaults
    df['Restaurant_Type'] = df['Restaurant_Type'].fillna('Unknown')
    df['Dish_Liked'] = df['Dish_Liked'].fillna('')
    df['Address'] = df['Address'].fillna('')
    df['Reviews_List'] = df['Reviews_List'].fillna('')
    
    missing_after = df.isnull().sum().sum()
    stats['missing_values_filled'] = missing_before - missing_after
    logger.info(f"Filled {missing_before - missing_after} missing values")
    
    # 12. Create additional useful columns
    # Extract city from location
    df['City'] = df['Location'].apply(lambda x: x.split(',')[0].strip() if ',' in str(x) else str(x).strip())
    
    # Create cuisine list for easier processing
    df['Cuisine_List'] = df['Cuisines'].apply(
        lambda x: [c.strip() for c in str(x).split(',') if c.strip()] if x else []
    )
    
    # Create price category
    def categorize_price(price):
        if price == 0:
            return 'Unknown'
        elif price < 300:
            return 'Budget'
        elif price < 600:
            return 'Moderate'
        elif price < 1000:
            return 'Expensive'
        else:
            return 'Very Expensive'
    
    df['Price_Category'] = df['Price'].apply(categorize_price)
    
    # Create rating category
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
            return 'Below Average'
    
    df['Rating_Category'] = df['Rating'].apply(categorize_rating)
    
    # 13. Sort by rating and votes for better recommendations
    df = df.sort_values(['Rating', 'Votes'], ascending=[False, False])
    
    # 14. Reset index
    df = df.reset_index(drop=True)
    
    # Final stats
    final_shape = df.shape
    stats['final_rows'] = final_shape[0]
    stats['final_columns'] = final_shape[1]
    
    logger.info(f"Final dataset shape: {final_shape}")
    logger.info(f"Total rows removed: {stats['rows_removed']}")
    logger.info(f"Total columns removed: {stats['columns_removed']}")
    logger.info(f"Missing values filled: {stats['missing_values_filled']}")
    
    # Save the cleaned dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"Cleaned dataset saved to {output_file}")
    
    return df, stats

def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the cleaned dataset
    
    Args:
        df: Cleaned dataframe
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_restaurants': len(df),
        'unique_cities': df['City'].nunique(),
        'unique_cuisines': len(set([cuisine for cuisine_list in df['Cuisine_List'] for cuisine in cuisine_list])),
        'rating_stats': {
            'mean': df['Rating'].mean(),
            'median': df['Rating'].median(),
            'min': df['Rating'].min(),
            'max': df['Rating'].max()
        },
        'price_stats': {
            'mean': df['Price'].mean(),
            'median': df['Price'].median(),
            'min': df['Price'].min(),
            'max': df['Price'].max()
        },
        'top_cities': df['City'].value_counts().head(10).to_dict(),
        'top_cuisines': pd.Series([cuisine for cuisine_list in df['Cuisine_List'] for cuisine in cuisine_list]).value_counts().head(10).to_dict()
    }
    
    return summary

if __name__ == "__main__":
    # Clean the dataset
    input_file = "../zomato.csv"
    output_file = "../data/zomato_cleaned.csv"
    
    try:
        df, stats = clean_restaurant_data(input_file, output_file)
        summary = get_data_summary(df)
        
        print("\n" + "="*50)
        print("DATA CLEANING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Original dataset: {stats['original_rows']} rows, {stats['original_columns']} columns")
        print(f"Cleaned dataset: {stats['final_rows']} rows, {stats['final_columns']} columns")
        print(f"Rows removed: {stats['rows_removed']}")
        print(f"Columns removed: {stats['columns_removed']}")
        print(f"Missing values filled: {stats['missing_values_filled']}")
        print("\nDataset Summary:")
        print(f"Total restaurants: {summary['total_restaurants']}")
        print(f"Unique cities: {summary['unique_cities']}")
        print(f"Unique cuisines: {summary['unique_cuisines']}")
        print(f"Average rating: {summary['rating_stats']['mean']:.2f}")
        print(f"Average price: Rs.{summary['price_stats']['mean']:.0f}")
        print("\nTop 5 Cities:")
        for city, count in list(summary['top_cities'].items())[:5]:
            print(f"  {city}: {count} restaurants")
        print("\nTop 5 Cuisines:")
        for cuisine, count in list(summary['top_cuisines'].items())[:5]:
            print(f"  {cuisine}: {count} restaurants")
        
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise
