#!/usr/bin/env python3
"""
Download required NLTK data for the ML recommendation engine
"""

import nltk
import ssl

def download_nltk_data():
    """Download required NLTK data"""
    try:
        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        print("Downloading NLTK data...")
        
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        print("NLTK data downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        return False

if __name__ == "__main__":
    success = download_nltk_data()
    exit(0 if success else 1)
