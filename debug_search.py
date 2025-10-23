import sys
sys.path.append('restaurant-recommendation')

from app import df, restaurants
import pandas as pd

print("=== Debug Search Issue ===")
print(f"Restaurants list length: {len(restaurants) if restaurants else 'None'}")
print(f"DataFrame df: {df is not None}")
if df is not None:
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"DataFrame head:")
    print(df.head())
    print(f"Combined_Features sample:")
    print(df["Combined_Features"].head() if "Combined_Features" in df.columns else "No Combined_Features column")
else:
    print("DataFrame is None!")

# Test search logic
if df is not None and not df.empty:
    query = "restaurant"
    mask = df["Combined_Features"].fillna("").str.contains(query, case=False, na=False)
    print(f"Search mask for '{query}': {mask.sum()} matches")
    if mask.sum() > 0:
        print("Sample matches:")
        print(df[mask][["Restaurant Name", "Cuisines", "City"]].head())
    else:
        print("No matches found")
        print("Sample Combined_Features values:")
        print(df["Combined_Features"].head())
