"""
Small training script to create a predictive typing model.
Saves model and vectorizer to `restaurant_recommender_model.joblib` in the project root.
Usage: python scripts/train_predictor.py dataset/restaurants_sample.csv
"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


def train(csv_path, out_path):
    df = pd.read_csv(csv_path)
    texts = df['name'].fillna('').astype(str).tolist()
    # Prepare labels: second token if present, else empty string
    X_texts = []
    y = []
    for t in texts:
        toks = [tok for tok in str(t).split() if tok]
        if len(toks) > 1:
            X_texts.append(t)
            y.append(toks[1])
    if not X_texts:
        print('No training pairs found (need names with at least two tokens).')
        return
    vect = CountVectorizer(ngram_range=(1,2), analyzer='char')
    X = vect.fit_transform(X_texts)
    model = LogisticRegression(max_iter=300)
    model.fit(X, y)
    joblib.dump({'vectorizer': vect, 'model': model}, out_path)
    print('Saved model to', out_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/train_predictor.py path/to/restaurants.csv')
        sys.exit(1)
    csv = sys.argv[1]
    out = Path('restaurant_predictor.joblib')
    train(csv, out)
