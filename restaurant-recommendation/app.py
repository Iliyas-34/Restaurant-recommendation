from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
from flask_caching import Cache
import os
import threading
import json
try:
    from flask_dance.contrib.google import make_google_blueprint, google  # type: ignore
    from flask_dance.contrib.facebook import make_facebook_blueprint, facebook  # type: ignore
    _OAUTH_AVAILABLE = True
except Exception:
    make_google_blueprint = None
    make_facebook_blueprint = None
    google = None
    facebook = None
    _OAUTH_AVAILABLE = False
import pandas as pd
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import csv
import os
from collections import defaultdict, Counter

# Import ML recommendation engine
try:
    from utils.ml_recommendation_engine import recommendation_engine, get_recommendations, initialize_recommendation_engine
    ML_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"ML recommendation engine not available: {e}")
    ML_ENGINE_AVAILABLE = False

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # change to a secure random key in production

# Configure Flask-Caching
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300  # 5 minutes
})
# ---------- OAuth Blueprints ----------
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID") or ""
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET") or ""
FACEBOOK_CLIENT_ID = os.environ.get("FACEBOOK_CLIENT_ID") or ""
FACEBOOK_CLIENT_SECRET = os.environ.get("FACEBOOK_CLIENT_SECRET") or ""

google_bp = None
facebook_bp = None
if _OAUTH_AVAILABLE and GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    google_bp = make_google_blueprint(
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scope=["profile", "email"],
        redirect_url="/login/google/callback"
    )
    app.register_blueprint(google_bp, url_prefix="/login")

if _OAUTH_AVAILABLE and FACEBOOK_CLIENT_ID and FACEBOOK_CLIENT_SECRET:
    facebook_bp = make_facebook_blueprint(
        client_id=FACEBOOK_CLIENT_ID,
        client_secret=FACEBOOK_CLIENT_SECRET,
        scope=["email"],
        redirect_url="/login/facebook/callback"
    )
    app.register_blueprint(facebook_bp, url_prefix="/login")


# ---------- File paths ----------
DATA_PATH = os.path.join("data", "restaurants.json")
WISHLIST_PATH = os.path.join("data", "wishlist.json")
USERS_PATH = os.path.join("data", "users.json")
FEEDBACK_PATH = os.path.join("data", "feedback.json")
REVIEWS_PATH = os.path.join("data", "reviews.json")
ORDERS_PATH = os.path.join("data", "orders.json")
PRICING_PATH = os.path.join("data", "pricing.json")


# ---------- Safe JSON helpers ----------
def load_json(path, default=None):
    if default is None:
        default = []
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_wishlist():
    return load_json(WISHLIST_PATH, [])


def save_wishlist(wishlist):
    save_json(WISHLIST_PATH, wishlist)


def load_users():
    return load_json(USERS_PATH, [])


def save_users(users):
    save_json(USERS_PATH, users)


def load_feedback():
    return load_json(FEEDBACK_PATH, [])


def save_feedback(feedback):
    save_json(FEEDBACK_PATH, feedback)


def load_reviews():
    return load_json(REVIEWS_PATH, [])


def save_reviews(reviews):
    save_json(REVIEWS_PATH, reviews)


def load_orders():
    return load_json(ORDERS_PATH, [])


def save_orders(orders):
    save_json(ORDERS_PATH, orders)


def load_pricing_data():
    """
    Load optional pricing dataset if present. Expected keys (flexible):
      - base_costs: { free: x, premium: y, enterprise: z }
      - multipliers: { per_restaurant: a, per_user: b }
      - features: { wishlist: c, reviews: d, admin: e }
    This is a heuristic helper to estimate pricing tiers.
    """
    default = {
        "base_costs": {"free": 0, "premium": 29, "enterprise": 99},
        "multipliers": {"per_restaurant": 0.02, "per_user": 0.05},
        "features": {"wishlist": 5, "reviews": 10, "admin": 15}
    }
    data = load_json(PRICING_PATH, default)
    # Merge defaults defensively
    for k in default:
        if k not in data or not isinstance(data[k], dict):
            data[k] = default[k]
        else:
            for sk, sv in default[k].items():
                data[k].setdefault(sk, sv)
    return data


# ---------- Load restaurants dataset ----------
# Primary dataset: Use Zomato data converted to JSON format
restaurants = []

try:
    # Load the converted JSON data (originally from zomato_cleaned.csv)
    json_path = os.path.join("data", "restaurants.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            restaurants = json.load(f)
        print(f"Loaded {len(restaurants)} restaurants from converted Zomato JSON dataset")
        print(f"Sample columns: {list(restaurants[0].keys()) if restaurants else 'No data'}")
    else:
        print("Converted Zomato JSON dataset not found!")
        restaurants = []
        
except Exception as e:
    print(f"Error loading converted Zomato JSON dataset: {e}")
    restaurants = []

# Defensive: ensure restaurants is a list of dicts
if not isinstance(restaurants, list):
    restaurants = []

print(f"Final dataset: {len(restaurants)} restaurants loaded")

# ---------- ML Recommendations setup using cleaned dataset ----------
def initialize_ml_data():
    """Initialize ML data and return the DataFrame"""
    global df, cosine_sim, tfidf, restaurants
    
    try:
        # Reload restaurants data if empty
        if not restaurants or len(restaurants) == 0:
            print("Reloading restaurants data...")
            try:
                # Load the processed data (preferably) or converted JSON data
                # Try multiple possible paths
                possible_paths = [
                    os.path.join("data", "restaurants_processed.json"),
                    os.path.join("data", "restaurants.json"),
                    os.path.join("restaurant-recommendation", "data", "restaurants_processed.json"),
                    os.path.join("restaurant-recommendation", "data", "restaurants.json"),
                    os.path.join("..", "restaurant-recommendation", "data", "restaurants_processed.json")
                ]
                
                json_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        json_path = path
                        break
                
                if json_path:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        restaurants = json.load(f)
                    print(f"Reloaded {len(restaurants)} restaurants from {json_path}")
                else:
                    print("Converted Zomato JSON dataset not found in any of the expected locations!")
                    print(f"Tried paths: {possible_paths}")
                    restaurants = []
            except Exception as e:
                print(f"Error reloading restaurants data: {e}")
                restaurants = []
        
        # Use the main restaurants dataset for ML
        df = pd.DataFrame(restaurants)
        print(f"Using main dataset with {len(df)} restaurants for ML")
        
        # Ensure required columns exist (using converted JSON column names)
        if "Cuisines" not in df.columns:
            df["Cuisines"] = ""
        df["Cuisines"] = df["Cuisines"].fillna("")
        
        if "City" not in df.columns:
            df["City"] = ""
        df["City"] = df["City"].fillna("")
        
        # Add missing columns with defaults
        if "Address" not in df.columns:
            df["Address"] = df["City"].astype(str) + ", " + df["Location"].astype(str)
        if "Latitude" not in df.columns:
            df["Latitude"] = 0.0
        if "Longitude" not in df.columns:
            df["Longitude"] = 0.0
        if "Price_Range" not in df.columns:
            df["Price_Range"] = df["Average Cost for two"].apply(lambda x: "₹" + str(x) if pd.notna(x) else "₹0")
        
        # Create combined features for better similarity (using converted JSON column names)
        df["Combined_Features"] = (
            df["Cuisines"].astype(str) + " " + 
            df["City"].astype(str) + " " + 
            df["Restaurant Name"].astype(str) + " " +
            df["Restaurant Type"].astype(str) + " " +
            df["Address"].astype(str)
        )
        
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 3))
        tfidf_matrix = tfidf.fit_transform(df["Combined_Features"])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print(f"ML similarity matrix created: {cosine_sim.shape}")
        
        return df
        
    except Exception as e:
        print(f"Error setting up ML recommendations: {e}")
        # If something goes wrong with pandas or ML setup, provide safe defaults
        df = pd.DataFrame(columns=["Restaurant_Name", "City", "Cuisines", "Rating", "Votes"])
        cosine_sim = None
        return df

# Initialize ML data
df = initialize_ml_data()

# Ensure data is loaded when Flask app starts
def load_data():
    global df, restaurants
    if not restaurants or len(restaurants) == 0:
        print("Loading data on first request...")
        df = initialize_ml_data()

# Load data immediately
load_data()
# ---------- Simple Predictive Typing Model (optional) ----------
try:
    # Prefer explicit dataset if present: dataset/restaurants.csv with column 'name'
    from pathlib import Path  # type: ignore
    suggest_vectorizer = None
    suggest_model = None
    _suggest_texts_for_fit = []
    _suggest_labels = []

    try:
        csv_path = Path(os.path.dirname(__file__)) / "dataset" / "restaurants.csv"
        if csv_path.exists():
            df_names = pd.read_csv(str(csv_path))
            names = df_names.get("name")
            if names is not None:
                names_list = names.dropna().astype(str).unique().tolist()
                _suggest_texts_for_fit = names_list  # use full names as training inputs (partial prefixes at inference)
                _suggest_labels = names_list        # predict the full name token
                print(f"Loaded {len(names_list)} names from CSV dataset for ML training: {names_list[:5]}...")
    except Exception as e:
        print(f"Error loading CSV dataset: {e}")
        pass

    # Fallback: derive from restaurants dataset (Restaurant_Name)
    if not _suggest_texts_for_fit and "Restaurant_Name" in df.columns:
        names_list = df["Restaurant_Name"].fillna("").astype(str).unique().tolist()
        _suggest_texts_for_fit = [n for n in names_list if n]
        _suggest_labels = [n for n in names_list if n]
        print(f"Using fallback dataset with {len(_suggest_texts_for_fit)} restaurant names")
except Exception:
    _suggest_labels = []
    _suggest_texts_for_fit = []
    suggest_vectorizer = None
    suggest_model = None

# Persisted predictor will be loaded after training in __main__ to preserve precedence


def recommend_restaurants(name, n=5):
    """
    Return up to `n` similar restaurants to `name`.
    If name not found or ML unavailable, return empty list.
    """
    if cosine_sim is None or df is None:
        return []
    
    try:
        # If name is a search term (not exact restaurant name), find best matches
        if name not in df["Restaurant Name"].values:
            # Use TF-IDF to find similar restaurants based on search term
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create search vector
            search_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 3))
            search_matrix = search_vectorizer.fit_transform(df["Combined_Features"])
            search_query = search_vectorizer.transform([name])
            
            # Calculate similarity
            search_similarity = cosine_similarity(search_query, search_matrix)
            sim_scores = list(enumerate(search_similarity[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[:n]  # Get top n matches
            restaurant_indices = [i[0] for i in sim_scores if i[1] > 0.1]  # Only include meaningful matches
        else:
            # Exact restaurant name match
            idx = int(df[df["Restaurant Name"] == name].index[0])
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n + 1]  # skip itself
            restaurant_indices = [i[0] for i in sim_scores]
    except Exception as e:
        print(f"Error in recommend_restaurants: {e}")
        return []
    
    # Return with appropriate column names (using converted JSON format)
    results = []
    for idx in restaurant_indices:
        restaurant = df.iloc[idx]
        result = {
            "name": restaurant.get("Restaurant Name", ""),
            "cuisines": restaurant.get("Cuisines", ""),
            "location": restaurant.get("Location", ""),
            "city": restaurant.get("City", ""),
            "address": restaurant.get("Address", ""),
            "rating": restaurant.get("Aggregate rating", 0),
            "cost": restaurant.get("Average Cost for two", 0),
            "votes": restaurant.get("Votes", 0),
            "type": restaurant.get("Restaurant Type", ""),
            "match_score": sim_scores[restaurant_indices.index(idx)][1] if restaurant_indices else 0
        }
        results.append(result)
    
    return results


# ---------- Categories (Cuisine-based) via simple clustering ----------
categories_cache = None

def compute_categories():
    global categories_cache
    try:
        if df is None or df.empty:
            categories_cache = {}
            return categories_cache

        work = df.copy()
        if "Cuisines" not in work.columns:
            work["Cuisines"] = ""
        if "Restaurant_Name" not in work.columns:
            work["Restaurant_Name"] = ""

        work["_rating"] = pd.to_numeric(work.get("Rating", 0), errors="coerce").fillna(0.0)
        work["_cost"] = pd.to_numeric(work.get("Price", 0), errors="coerce").fillna(0.0)
        cuisines_series = work["Cuisines"].fillna("").astype(str).apply(lambda s: s.split(",")[0].strip() if s else "General")
        unique_cuisines = [c for c in cuisines_series.unique() if c]

        # Fallback: group directly by cuisine if clustering is not meaningful
        if len(work) < 5 or work[["_rating", "_cost"]].nunique().sum() <= 1 or not unique_cuisines:
            grouped = {}
            for _, row in work.iterrows():
                cat = (row.get("Cuisines") or "").split(",")[0].strip() or "General"
                grouped.setdefault(cat, []).append(row.get("Restaurant_Name") or "")
            categories_cache = {k: sorted([n for n in v if n]) for k, v in grouped.items()}
            return categories_cache

        n_clusters = max(2, min(len(unique_cuisines), 15))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        X = work[["_rating", "_cost"]].to_numpy()
        labels_idx = km.fit_predict(X)
        work["_cluster"] = labels_idx
        work["_primary_cuisine"] = cuisines_series

        # Map cluster -> majority cuisine label
        cluster_to_cuisine = {}
        for cid in sorted(work["_cluster"].unique()):
            subset = work[work["_cluster"] == cid]
            top = subset["_primary_cuisine"].value_counts().idxmax() if not subset.empty else "General"
            cluster_to_cuisine[cid] = top

        grouped = {}
        for _, row in work.iterrows():
            cat = cluster_to_cuisine.get(int(row["_cluster"]), "General")
            name = row.get("Restaurant_Name") or ""
            if name:
                grouped.setdefault(cat, []).append(name)

        categories_cache = dict(sorted({k: sorted(v) for k, v in grouped.items()}.items(), key=lambda kv: kv[0]))
        return categories_cache
    except Exception:
        categories_cache = {}
        return categories_cache

# ---------- Routes ----------
@app.route("/")
def homepage():
    return render_template("home.html")


@app.route("/restaurants", methods=["GET"])  # UI still uses frontend fetch for listing
def restaurants_page():
    # Keep server rendering minimal; search param will be used by frontend fetch as well
    return render_template("restaurant.html")

@app.route("/restaurant/<restaurant_name>")
def restaurant_details_page(restaurant_name):
    """Individual restaurant details page"""
    try:
        global restaurants
        if not restaurants or len(restaurants) == 0:
            return render_template("restaurant_details.html", error="Restaurant data not available")
        
        # Find restaurant by name
        restaurant = None
        for r in restaurants:
            if r.get("Restaurant Name") == restaurant_name:
                restaurant = r
                break
        
        if not restaurant:
            return render_template("restaurant_details.html", error="Restaurant not found")
        
        # Get similar restaurants
        similar_restaurants = get_similar_restaurants(restaurant_name, limit=6)
        
        # Format restaurant data for template
        restaurant_data = {
            "id": restaurant.get("Restaurant Name", ""),
            "name": restaurant.get("Restaurant Name", ""),
            "cuisines": restaurant.get("Cuisines", ""),
            "city": restaurant.get("City", ""),
            "location": restaurant.get("Location", ""),
            "address": restaurant.get("Address", ""),
            "rating": float(restaurant.get("Aggregate rating", 0)),
            "votes": int(restaurant.get("Votes", 0)),
            "price": float(restaurant.get("Average Cost for two", 0)),
            "price_range": restaurant.get("Price_Range", "₹0"),
            "type": restaurant.get("Restaurant Type", ""),
            "latitude": float(restaurant.get("Latitude", 0)) if restaurant.get("Latitude") else None,
            "longitude": float(restaurant.get("Longitude", 0)) if restaurant.get("Longitude") else None,
            "online_order": restaurant.get("Online Order", "No"),
            "book_table": restaurant.get("Book Table", "No"),
            "dish_liked": restaurant.get("Dish Liked", ""),
            "reviews_list": restaurant.get("Reviews List", ""),
            "cuisine_list": restaurant.get("Cuisine List", ""),
            "price_category": restaurant.get("Price Category", ""),
            "rating_category": restaurant.get("Rating Category", "")
        }
        
        return render_template("restaurant_details.html", 
                             restaurant=restaurant_data, 
                             similar_restaurants=similar_restaurants)
    except Exception as e:
        print(f"Error in restaurant_details_page: {e}")
        return render_template("restaurant_details.html", error="Failed to load restaurant details")


@app.route("/wishlist")
def wishlist_page():
    return render_template("wishlist.html")


@app.route("/about")
def about_page():
    return render_template("about.html")


# ---------- Contact & Feedback ----------
@app.route("/contact-feedback", methods=["GET", "POST"])
def contact_feedback_page():
    if request.method == "POST":
        form_type = request.form.get("form_type")
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()

        feedback = load_feedback()

        entry = {
            "type": form_type or "contact",
            "name": name,
            "email": email,
            "message": message,
            "rating": None,
            "date": datetime.datetime.now().strftime("%b %d, %Y - %I:%M %p")
        }

        if form_type == "feedback":
            entry["rating"] = request.form.get("rating")

        feedback.append(entry)
        save_feedback(feedback)

        if form_type == "feedback":
            flash("✅ Thank you for your feedback!", "success")
        else:
            flash("📩 Your message has been sent! Thank you for contacting us.", "success")

        return redirect(url_for("contact_feedback_page"))

    # only show feedback entries (not contact messages)
    feedback_list = [f for f in load_feedback() if f.get("type") == "feedback"]
    feedback_list = sorted(feedback_list, key=lambda x: x.get("date", ""), reverse=True)
    return render_template("contact_feedback.html", feedback=feedback_list)


# ---------- Admin: Reviews Management ----------
@app.route("/admin/reviews", methods=["GET"])
def admin_reviews_page():
    """
    Simple admin listing of user reviews with delete buttons.
    Reviews are stored in reviews.json with fields: id, username, text, rating, date
    """
    reviews = load_reviews()
    # defensive defaults
    for r in reviews:
        r.setdefault("id", str(r.get("id") or r.get("_id") or ""))
        r.setdefault("username", r.get("username") or r.get("name") or "Anonymous")
        r.setdefault("text", r.get("text") or r.get("message") or "")
        r.setdefault("rating", r.get("rating") or "")
        r.setdefault("date", r.get("date") or "")
    return render_template("admin_reviews.html", reviews=reviews)


@app.route("/review/<rid>", methods=["DELETE"])
def delete_review(rid):
    if not rid:
        return jsonify({"message": "Missing review id"}), 400
    reviews = load_reviews()
    before_count = len(reviews)
    reviews = [r for r in reviews if str(r.get("id")) != str(rid)]
    if len(reviews) == before_count:
        return jsonify({"message": "Review not found"}), 404
    save_reviews(reviews)
    return jsonify({"message": "Deleted"})


# ---------- Auth (merged login/signup) ----------
@app.route("/auth", methods=["GET", "POST"])
def auth_page():
    users = load_users()
    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "login":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            user = next((u for u in users if u.get("username") == username and u.get("password") == password), None)
            if user:
                session["user"] = username
                flash("✅ Login successful!", "success")
                return redirect(url_for("homepage"))
            else:
                flash("❌ Invalid username or password", "error")

        elif form_type == "signup":
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip()
            password = request.form.get("password", "")

            if not username or not password:
                flash("⚠️ Username and password are required", "error")
                return redirect(url_for("auth_page"))
            if any(u.get("username") == username for u in users):
                flash("⚠️ Username already taken!", "error")
                return redirect(url_for("auth_page"))

            users.append({"username": username, "email": email, "password": password})
            save_users(users)
            # Auto-login and redirect to profile
            session["user"] = username
            session["username"] = username
            flash("✅ Signup successful! Welcome, " + username + ".", "success")
            return redirect(url_for("profile_page"))

        return redirect(url_for("auth_page"))

    return render_template("auth.html", oauth_available=_OAUTH_AVAILABLE, google_enabled=bool(google_bp), facebook_enabled=bool(facebook_bp))


# ---------- Split Auth Pages (UI only) ----------
@app.route("/auth/login", methods=["GET"])
def login_page():
    return render_template("login.html")


@app.route("/auth/signup", methods=["GET"])
def signup_page():
    return render_template("signup.html")


# ---------- OAuth Routes ----------
@app.route("/login/google")
def login_google():
    if not (_OAUTH_AVAILABLE and google_bp):
        flash("Google login not configured on this server.", "error")
        return redirect(url_for("auth_page"))
    if not google.authorized:
        return redirect(url_for("google.login"))
    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        flash("Google login failed", "error")
        return redirect(url_for("auth_page"))
    data = resp.json() or {}
    email = data.get("email")
    username = (data.get("name") or (email.split("@")[0] if email else "google_user")).strip()

    # Ensure user exists in users.json
    users = load_users()
    user = next((u for u in users if u.get("username") == username), None)
    if not user:
        users.append({"username": username, "email": email or "", "password": ""})
        save_users(users)
    session["user"] = username
    session["username"] = username
    return redirect(url_for("profile_page"))


@app.route("/login/facebook")
def login_facebook():
    if not (_OAUTH_AVAILABLE and facebook_bp):
        flash("Facebook login not configured on this server.", "error")
        return redirect(url_for("auth_page"))
    if not facebook.authorized:
        return redirect(url_for("facebook.login"))
    resp = facebook.get("/me?fields=id,name,email")
    if not resp.ok:
        flash("Facebook login failed", "error")
        return redirect(url_for("auth_page"))
    data = resp.json() or {}
    email = data.get("email")
    username = (data.get("name") or (email.split("@")[0] if email else "fb_user")).strip()

    users = load_users()
    user = next((u for u in users if u.get("username") == username), None)
    if not user:
        users.append({"username": username, "email": email or "", "password": ""})
        save_users(users)
    session["user"] = username
    session["username"] = username
    return redirect(url_for("profile_page"))


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("ℹ️ You have been logged out.", "info")
    return redirect(url_for("homepage"))


# ---------- Profile ----------
@app.route("/profile", methods=["GET"])
def profile_page():
    username = session.get("user")
    users = load_users()
    user = next((u for u in users if u.get("username") == username), None)
    # Provide safe defaults
    profile = {
        "username": user.get("username") if user else "Guest",
        "email": user.get("email") if user else "",
        "avatar_url": "https://ui-avatars.com/api/?name=" + (user.get("username") if user else "Guest")
    }
    return render_template("profile.html", profile=profile)


@app.route("/profile/edit", methods=["POST"])
def edit_profile():
    username = session.get("user")
    if not username:
        return redirect(url_for("auth_page"))
    users = load_users()
    user = next((u for u in users if u.get("username") == username), None)
    if not user:
        flash("User not found", "error")
        return redirect(url_for("profile_page"))

    new_username = (request.form.get("username") or username).strip()
    new_email = (request.form.get("email") or user.get("email", "")).strip()
    new_password = request.form.get("password") or user.get("password")

    # If username changed and collides, prevent
    if new_username != username and any(u.get("username") == new_username for u in users):
        flash("⚠️ Username already taken!", "error")
        return redirect(url_for("profile_page"))

    user["username"] = new_username
    user["email"] = new_email
    user["password"] = new_password
    save_users(users)

    # Update session if username changed
    session["user"] = new_username
    flash("✅ Profile updated", "success")
    return redirect(url_for("profile_page"))


# ---------- User Orders ----------
@app.route("/user/orders", methods=["GET"])
def user_orders():
    username = session.get("user")
    if not username:
        return jsonify([])
    orders = load_orders()
    # Expect order: { id, username, restaurant, items:[...], date, total }
    user_orders = [o for o in orders if o.get("username") == username]
    return jsonify(user_orders)


# ---------- User Wishlist (read-only) ----------
@app.route("/user/wishlist", methods=["GET"])
def user_wishlist():
    username = session.get("user") or ""
    wl = load_wishlist()
    items = []
    for item in wl:
        name = item.get("name") or item.get("Restaurant_Name")
        iuser = item.get("username") or ""
        if name and iuser == username:
            items.append({"id": name, "name": name})
    return jsonify(items)


# ---------- Wishlist Delete by id (name) ----------
@app.route("/wishlist/<id>", methods=["DELETE"])
def delete_wishlist_item(id):
    wishlist = load_wishlist()
    new_list = [item for item in wishlist if (item.get("name") or item.get("Restaurant_Name")) != id]
    if len(new_list) == len(wishlist):
        return jsonify({"message": "Not found"}), 404
    save_wishlist(new_list)
    return jsonify({"message": "Removed"})


# ---------- POST Logout (keep GET for backward compatibility) ----------
@app.route("/logout", methods=["POST"])  # extend existing endpoint to support POST
def logout_post():
    session.pop("user", None)
    return jsonify({"message": "logged_out"})


# ---------- Dedicated Signup Endpoint ----------
@app.route("/signup", methods=["POST"])
def signup_post():
    users = load_users()
    username = (request.form.get("username") or (request.json or {}).get("username") or "").strip()
    email = (request.form.get("email") or (request.json or {}).get("email") or "").strip()
    password = (request.form.get("password") or (request.json or {}).get("password") or "")

    if not username or not password:
        # Support both API and form usage
        if request.is_json:
            return jsonify({"message": "Username and password are required"}), 400
        flash("⚠️ Username and password are required", "error")
        return redirect(url_for("auth_page"))

    if any(u.get("username") == username for u in users):
        if request.is_json:
            return jsonify({"message": "Username already taken"}), 409
        flash("⚠️ Username already taken!", "error")
        return redirect(url_for("auth_page"))

    users.append({"username": username, "email": email, "password": password})
    save_users(users)

    session["user"] = username
    session["username"] = username

    # JSON clients get JSON; form clients redirect
    if request.is_json:
        return jsonify({"message": "signup_success", "username": username}), 201
    flash("✅ Signup successful! Welcome, " + username + ".", "success")
    return redirect(url_for("profile_page"))


# ---------- API: Restaurants ----------
@app.route("/api/restaurants")
def get_restaurants():
    """
    Query string:
      - search (substring on Restaurant_Name)
      - city (single value or comma-separated list)
      - cuisine (single value or comma-separated list; matching is case-insensitive substring)
      - rating (min aggregate rating as number)
      - sort (rating|votes|cost_low|cost_high)
      - page (pagination, 1-based)
    """
    global restaurants
    
    # Debug: Check restaurants data
    print(f"Debug: restaurants type: {type(restaurants)}, length: {len(restaurants) if restaurants else 0}")
    if restaurants and len(restaurants) > 0:
        print(f"Debug: First restaurant keys: {list(restaurants[0].keys()) if restaurants[0] else 'Empty'}")
        print(f"Debug: First restaurant name: '{restaurants[0].get('Restaurant Name', 'MISSING')}'")
    
    search = request.args.get("search", "").strip().lower()
    cities_raw = request.args.get("city", "").strip()
    cuisines_raw = request.args.get("cuisine", "").strip()
    rating = request.args.get("rating", "").strip()
    sort = request.args.get("sort", "").strip()
    page = int(request.args.get("page", 1) or 1)
    per_page = 20

    # parse multi-values: allow both single value and comma-separated lists
    city_list = [c.strip() for c in cities_raw.split(",") if c.strip()] if cities_raw else []
    cuisine_list = [c.strip().lower() for c in cuisines_raw.split(",") if c.strip()] if cuisines_raw else []

    # Ensure we have valid restaurants data
    if not restaurants or len(restaurants) == 0:
        print("Debug: No restaurants data available, reloading...")
        try:
            json_path = os.path.join("data", "restaurants.json")
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    restaurants = json.load(f)
                print(f"Debug: Reloaded {len(restaurants)} restaurants")
            else:
                print("Debug: restaurants.json not found!")
                return jsonify({"restaurants": [], "total": 0, "page": page, "per_page": per_page})
        except Exception as e:
            print(f"Debug: Error reloading restaurants: {e}")
            return jsonify({"restaurants": [], "total": 0, "page": page, "per_page": per_page})

    filtered = restaurants

    if search:
        # Search in restaurant name, cuisines, and city
        search_lower = search.lower()
        filtered = [r for r in filtered if (
            search_lower in str(r.get("Restaurant Name", "")).lower() or
            search_lower in str(r.get("Cuisines", "")).lower() or
            search_lower in str(r.get("City", "")).lower() or
            search_lower in str(r.get("Dish Liked", "")).lower()
        )]

    if city_list:
        # exact match on City field (case-sensitive as stored); normalize both sides if you prefer
        filtered = [r for r in filtered if str(r.get("City", "")).strip() in city_list]

    if cuisine_list:
        def cuisine_matches(r):
            cuisines_field = str(r.get("Cuisines", "")).lower()
            return any(c in cuisines_field for c in cuisine_list)
        filtered = [r for r in filtered if cuisine_matches(r)]

    if rating:
        try:
            min_rating = float(rating)
            filtered = [r for r in filtered if float(r.get("Aggregate rating", 0) or 0) >= min_rating]
        except Exception:
            pass

    # Sorting (defensive numeric conversions)
    try:
        if sort == "rating":
            filtered = sorted(filtered, key=lambda r: float(r.get("Aggregate rating", 0) or 0), reverse=True)
        elif sort == "votes":
            filtered = sorted(filtered, key=lambda r: int(float(r.get("Votes", 0) or 0)), reverse=True)
        elif sort == "cost_low":
            filtered = sorted(filtered, key=lambda r: int(float(r.get("Average Cost for two", 0) or 0)))
        elif sort == "cost_high":
            filtered = sorted(filtered, key=lambda r: int(float(r.get("Average Cost for two", 0) or 0)), reverse=True)
    except Exception:
        # if conversion fails for any record, skip sort
        pass

    total = len(filtered)
    start = (page - 1) * per_page
    end = start + per_page
    paginated = filtered[start:end]

    # Format results for frontend
    formatted_restaurants = []
    for restaurant in paginated:
        formatted_restaurants.append({
            "id": restaurant.get("Restaurant Name", ""),
            "name": restaurant.get("Restaurant Name", ""),
            "cuisines": restaurant.get("Cuisines", ""),
            "city": restaurant.get("City", ""),
            "location": restaurant.get("Location", ""),
            "address": restaurant.get("Address", ""),
            "rating": float(restaurant.get("Aggregate rating", 0)),
            "votes": int(restaurant.get("Votes", 0)),
            "price": float(restaurant.get("Average Cost for two", 0)),
            "price_range": f"₹{restaurant.get('Average Cost for two', 0)}",
            "type": restaurant.get("Restaurant Type", ""),
            "latitude": float(restaurant.get("Latitude", 0)) if restaurant.get("Latitude") else None,
            "longitude": float(restaurant.get("Longitude", 0)) if restaurant.get("Longitude") else None,
            "online_order": restaurant.get("Online Order", "No"),
            "book_table": restaurant.get("Book Table", "No")
        })

    return jsonify({
        "restaurants": formatted_restaurants,
        "total": total,
        "page": page,
        "per_page": per_page
    })


# ---------- API: Filters (cities + cuisines) ----------
@app.route("/api/filters")
def get_filters():
    """
    Returns JSON:
      { "cities": [...], "cuisines": [...] }
    Cities and cuisines are cleaned strings (trimmed). Cuisines are split on comma.
    """
    # Unique cities (cleaned)
    cities_set = set()
    for r in restaurants:
        city_val = r.get("City")
        if city_val:
            city_str = str(city_val).strip()
            if city_str:
                cities_set.add(city_str)

    # Unique cuisines (split by comma, cleaned)
    cuisines_set = set()
    for r in restaurants:
        c_field = r.get("Cuisines")
        if c_field:
            for part in str(c_field).split(","):
                cs = part.strip()
                if cs:
                    cuisines_set.add(cs)

    cities = sorted(cities_set)
    cuisines = sorted(cuisines_set)

    return jsonify({"cities": cities, "cuisines": cuisines})


# ---------- Pages: Pricing & Info ----------
@app.route("/pricing")
def pricing_page():
    pricing_data = load_pricing_data()
    # Derive metrics from existing datasets (fallbacks safe)
    num_restaurants = len(restaurants or [])
    num_users = len(load_users() or [])
    features_enabled = {
        "wishlist": True,
        "reviews": len(load_reviews() or []) > 0,  # consider reviews feature if any review exists
        "admin": True
    }

    # Simple heuristic pricing estimator
    base = pricing_data.get("base_costs", {})
    mult = pricing_data.get("multipliers", {})
    feat = pricing_data.get("features", {})

    def estimate(tier):
        cost = float(base.get(tier, 0))
        cost += float(mult.get("per_restaurant", 0)) * num_restaurants
        cost += float(mult.get("per_user", 0)) * num_users
        for fname, enabled in features_enabled.items():
            if enabled:
                cost += float(feat.get(fname, 0))
        return round(cost, 2)

    tiers = {
        "free": max(0, estimate("free")),
        "premium": max(0, estimate("premium")),
        "enterprise": max(0, estimate("enterprise"))
    }

    return render_template("pricing.html", tiers=tiers, metrics={
        "num_restaurants": num_restaurants,
        "num_users": num_users,
        "features": features_enabled
    })


@app.route("/tutorials")
def tutorials_page():
    return render_template("tutorials.html")


@app.route("/docs")
def docs_page():
    return render_template("docs.html")


@app.route("/careers")
def careers_page():
    return render_template("careers.html")

# Alias endpoint to match expected name
@app.route("/careers", endpoint="careers")
def careers():
    return render_template("careers.html")


@app.route("/faq")
def faq_page():
    return render_template("faq.html")


@app.route("/terms")
def terms_page():
    return render_template("terms.html")


@app.route("/privacy")
def privacy_page():
    return render_template("privacy.html")


@app.route("/cookies")
def cookies_page():
    return render_template("cookies.html")


# ---------- Category Redirect (reuses existing restaurants page & filters) ----------
@app.route("/category/<name>")
def category_redirect(name):
    # Redirect to restaurants page with cuisine filter applied, preserving existing logic
    return redirect(url_for("restaurants_page") + f"?cuisine={request.view_args.get('name')}")


# ---------- Predictive Typing Endpoint ----------
@app.route("/api/suggestions", methods=["GET"])
def get_search_suggestions():
    """Get search suggestions based on partial text"""
    try:
        query = request.args.get("q", "").strip().lower()
        limit = int(request.args.get("limit", 10))
        
        if len(query) < 2:
            return jsonify({"suggestions": []})
        
        suggestions = []
        
        # Get restaurant name suggestions
        for restaurant in restaurants:
            name = str(restaurant.get("Restaurant Name", "")).lower()
            cuisines = str(restaurant.get("Cuisines", "")).lower()
            city = str(restaurant.get("City", "")).lower()
            
            if query in name:
                suggestions.append({
                    "text": restaurant.get("Restaurant Name", ""),
                    "type": "restaurant",
                    "city": restaurant.get("City", ""),
                    "cuisines": restaurant.get("Cuisines", "")
                })
            elif query in cuisines:
                # Extract cuisine suggestions
                cuisine_list = [c.strip() for c in cuisines.split(",")]
                for cuisine in cuisine_list:
                    if query in cuisine.lower() and cuisine not in [s["text"] for s in suggestions]:
                        suggestions.append({
                            "text": cuisine.strip(),
                            "type": "cuisine"
                        })
            elif query in city:
                if city not in [s["text"] for s in suggestions]:
                    suggestions.append({
                        "text": restaurant.get("City", ""),
                        "type": "city"
                    })
            
            if len(suggestions) >= limit:
                break
        
        return jsonify({"suggestions": suggestions[:limit]})
        
    except Exception as e:
        print(f"Error getting suggestions: {e}")
        return jsonify({"suggestions": []})


def predict_next_token():
    try:
        data = request.get_json(force=True) or {}
        text = str(data.get("text") or "")
    except Exception:
        text = ""
    
    print(f"Prediction request for: '{text}'")
    print(f"Model status - vectorizer: {suggest_vectorizer is not None}, model: {suggest_model is not None}")
    
    if not text or suggest_vectorizer is None or suggest_model is None:
        print("Model not ready, returning empty suggestion")
        return jsonify({"suggestion": "", "next_token": ""})
    
    try:
        # Log user search interaction (DISABLED)
        # log_user_interaction('search', {'query': text})
        
        vec = suggest_vectorizer.transform([text])
        pred = suggest_model.predict(vec)
        suggestion = str(pred[0])
        print(f"Prediction result: '{suggestion}'")
        return jsonify({"suggestion": suggestion, "next_token": suggestion})
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"suggestion": "", "next_token": ""})


# Test endpoint to check model status
@app.route("/predict/status", methods=["GET"])
def predict_status():
    return jsonify({
        "vectorizer_ready": suggest_vectorizer is not None,
        "model_ready": suggest_model is not None,
        "training_data_count": len(_suggest_texts_for_fit) if '_suggest_texts_for_fit' in globals() else 0,
        "sample_data": _suggest_texts_for_fit[:5] if '_suggest_texts_for_fit' in globals() and _suggest_texts_for_fit else []
    })


# ---------- Chatbot Functionality ----------
chatbot_vectorizer = None
chatbot_vectors = None
chatbot_items = []
chatbot_descriptions = []

def initialize_chatbot():
    global chatbot_vectorizer, chatbot_vectors, chatbot_items, chatbot_descriptions
    try:
        print("Initializing chatbot with restaurant data...")
        print(f"Total restaurants available: {len(restaurants)}")
        
        # Use restaurant data for chatbot - create comprehensive descriptions
        restaurant_data = []
        descriptions = []
        
        for r in restaurants:
            # Use actual column names from the loaded data
            name = r.get("Restaurant Name", "")
            cuisine = r.get("Cuisines", "")
            city = r.get("City", "")
            country = r.get("Country_Name", "")
            rating = r.get("Aggregate rating", "")
            cost = r.get("Average Cost for two", "")
            
            if name:  # Only include restaurants with names
                # Create comprehensive description for better matching
                desc_parts = [name]
                if cuisine and cuisine != "Unknown" and cuisine != "":
                    desc_parts.append(cuisine)
                if city and city != "":
                    desc_parts.append(city)
                if country and country != "":
                    desc_parts.append(country)
                if rating and rating != "":
                    desc_parts.append(f"rating {rating}")
                if cost and cost != "":
                    desc_parts.append(f"cost {cost}")
                
                combined_desc = " ".join(desc_parts)
                restaurant_data.append({
                    'name': name,
                    'cuisine': cuisine,
                    'city': city,
                    'country': country,
                    'rating': rating,
                    'cost': cost,
                    'description': combined_desc
                })
                descriptions.append(combined_desc)
        
        if not restaurant_data:
            print("No restaurant data found for chatbot initialization")
            print(f"Total restaurants available: {len(restaurants)}")
            print(f"Sample restaurant keys: {list(restaurants[0].keys()) if restaurants else 'No restaurants'}")
            chatbot_vectorizer = None
            chatbot_vectors = None
            chatbot_items = []
            chatbot_descriptions = []
            return
        
        chatbot_items = restaurant_data
        chatbot_descriptions = descriptions
        
        # Train TF-IDF vectorizer with more features for better matching
        chatbot_vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=3000,
            ngram_range=(1, 3),  # Include trigrams for better matching
            min_df=1,
            max_df=0.95
        )
        chatbot_vectors = chatbot_vectorizer.fit_transform(descriptions)
        
        print(f"Chatbot initialized with {len(chatbot_items)} restaurants")
        print(f"Vectorizer features: {chatbot_vectorizer.get_feature_names_out().shape[0]}")
        print(f"Sample restaurants: {[r['name'] for r in restaurant_data[:5]]}")
        
        # Test the chatbot with a sample query
        test_query = "pizza"
        if chatbot_vectorizer and chatbot_vectors is not None:
            test_vec = chatbot_vectorizer.transform([test_query])
            test_scores = cosine_similarity(test_vec, chatbot_vectors)
            top_matches = test_scores[0].argsort()[-3:][::-1]
            print(f"Test query '{test_query}' top matches: {[chatbot_items[i]['name'] for i in top_matches if test_scores[0][i] > 0.01]}")
        
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        import traceback
        traceback.print_exc()
        chatbot_vectorizer = None
        chatbot_vectors = None
        chatbot_items = []
        chatbot_descriptions = []

def get_chatbot_response(query):
    if chatbot_vectorizer is None or chatbot_vectors is None:
        return "I'm still learning about our restaurants. Please try again later!"
    
    try:
        print(f"Processing chatbot query: '{query}'")
        
        # Transform query
        query_vec = chatbot_vectorizer.transform([query])
        
        # Calculate similarity
        scores = cosine_similarity(query_vec, chatbot_vectors)
        top_indices = scores[0].argsort()[-10:][::-1]  # Get top 10 matches
        
        # Filter results with meaningful similarity (lower threshold for better results)
        good_matches = [(idx, scores[0][idx]) for idx in top_indices if scores[0][idx] > 0.01]
        
        print(f"Found {len(good_matches)} matches with similarity > 0.01")
        
        if not good_matches:
            # Fallback: try to find partial matches with more flexible matching
            query_lower = query.lower()
            fallback_matches = []
            
            # Split query into words for better matching
            query_words = query_lower.split()
            
            for i, restaurant in enumerate(chatbot_items):
                name_lower = restaurant['name'].lower()
                cuisine_lower = restaurant.get('cuisine', '').lower()
                city_lower = restaurant.get('city', '').lower()
                description_lower = restaurant.get('description', '').lower()
                
                # Check for any word matches
                match_score = 0
                for word in query_words:
                    if (word in name_lower or 
                        word in cuisine_lower or 
                        word in city_lower or
                        word in description_lower):
                        match_score += 0.2
                
                if match_score > 0:
                    fallback_matches.append((i, match_score))
            
            # Sort by match score
            fallback_matches.sort(key=lambda x: x[1], reverse=True)
            good_matches = fallback_matches[:5]
            print(f"Found {len(good_matches)} fallback matches")
        
        if good_matches:
            response = "Here are some restaurants that match your query:\n\n"
            for i, (idx, score) in enumerate(good_matches[:5]):
                restaurant = chatbot_items[idx]
                name = restaurant['name']
                cuisine = restaurant.get('cuisine', 'Unknown cuisine')
                city = restaurant.get('city', 'Unknown city')
                rating = restaurant.get('rating', 'N/A')
                cost = restaurant.get('cost', 'N/A')
                
                response += f"**{i+1}. {name}**\n"
                response += f"   Cuisine: {cuisine}\n"
                response += f"   Location: {city}\n"
                if rating != 'N/A' and rating != '':
                    response += f"   Rating: {rating}/5\n"
                if cost != 'N/A' and cost != '':
                    response += f"   Average Cost for Two: ${cost}\n"
                response += "\n"
        else:
            response = "I couldn't find specific restaurants matching your query. Try asking about:\n"
            response += "• Specific cuisines (pizza, sushi, Chinese, Italian, etc.)\n"
            response += "• Restaurant names\n"
            response += "• Cities or locations\n"
            response += "• Food types (burgers, pasta, etc.)\n"
            response += "• Popular dishes (ramen, tacos, burgers, etc.)"
        
        print(f"Chatbot response generated: {len(response)} characters")
        return response
    except Exception as e:
        print(f"Error in chatbot response: {e}")
        import traceback
        traceback.print_exc()
        return "Sorry, I encountered an error. Please try again!"

@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        data = request.get_json(force=True) or {}
        query = str(data.get("message", "")).strip()
        
        print(f"Chatbot received query: '{query}'")
        print(f"Chatbot vectorizer ready: {chatbot_vectorizer is not None}")
        print(f"Chatbot vectors ready: {chatbot_vectors is not None}")
        print(f"Chatbot items count: {len(chatbot_items)}")
        
        if not query:
            return jsonify({"response": "Please ask me about any food or restaurant!"})
        
        # Log chatbot interaction (DISABLED)
        # log_user_interaction('chatbot', {'query': query})
        
        if chatbot_vectorizer is None or chatbot_vectors is None:
            print("Chatbot not initialized yet")
            return jsonify({"response": "I'm still learning about our restaurants. Please wait a moment and try again!"})
        
        response = get_chatbot_response(query)
        print(f"Chatbot response: '{response[:100]}...'")
        return jsonify({"response": response})
    except Exception as e:
        print(f"Chatbot error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"response": "Sorry, I encountered an error. Please try again later!"})

# Test endpoint to check chatbot status
@app.route("/chatbot/status", methods=["GET"])
def chatbot_status():
    return jsonify({
        "vectorizer_ready": chatbot_vectorizer is not None,
        "vectors_ready": chatbot_vectors is not None,
        "restaurant_count": len(chatbot_items),
        "sample_restaurants": [r['name'] for r in chatbot_items[:5]] if chatbot_items else []
    })

# Test endpoint to manually test chatbot
@app.route("/chatbot/test", methods=["GET"])
def chatbot_test():
    test_query = "pizza"
    try:
        if chatbot_vectorizer is None or chatbot_vectors is None:
            return jsonify({"error": "Chatbot not initialized", "vectorizer_ready": chatbot_vectorizer is not None, "vectors_ready": chatbot_vectors is not None})
        
        response = get_chatbot_response(test_query)
        return jsonify({
            "test_query": test_query,
            "response": response,
            "vectorizer_ready": chatbot_vectorizer is not None,
            "vectors_ready": chatbot_vectors is not None,
            "restaurant_count": len(chatbot_items)
        })
    except Exception as e:
        return jsonify({"error": str(e), "vectorizer_ready": chatbot_vectorizer is not None, "vectors_ready": chatbot_vectors is not None})


# ---------- ML Category Extraction ----------
def extract_categories_from_dataset():
    """Extract top categories from restaurant dataset using ML"""
    try:
        print("Extracting categories from restaurant dataset...")
        print(f"Processing {len(restaurants)} restaurants")
        
        # Extract cuisines from restaurant data
        cuisines = []
        for restaurant in restaurants:
            # Use JSON format (full restaurant data)
            cuisine = restaurant.get("Cuisines", "")
            if cuisine and cuisine != "Unknown" and cuisine != "":
                # Split multiple cuisines and clean them
                cuisine_list = [c.strip() for c in str(cuisine).split(',') if c.strip()]
                cuisines.extend(cuisine_list)
        
        if not cuisines:
            print("No cuisines found in dataset")
            return ["Pizza", "Chinese", "Italian", "Japanese", "Indian", "Mexican", "Thai", "American"]
        
        print(f"Found {len(cuisines)} cuisine entries")
        
        # Use CountVectorizer to get cuisine frequency
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(
            lowercase=True,
            stop_words='english',
            min_df=1,
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ characters
        )
        
        X = vectorizer.fit_transform(cuisines)
        feature_names = vectorizer.get_feature_names_out()
        frequencies = X.sum(axis=0).A1  # Convert to 1D array
        
        # Create frequency dictionary
        category_freq = dict(zip(feature_names, frequencies))
        
        # Sort by frequency and get top categories
        top_categories = sorted(category_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out very common words and get top 8 categories
        filtered_categories = []
        common_words = {'food', 'restaurant', 'cafe', 'bar', 'grill', 'kitchen', 'house', 'place', 'restaurants', 'cafes'}
        
        for category, freq in top_categories:
            if (category not in common_words and 
                len(category) > 2 and 
                freq > 1 and 
                len(filtered_categories) < 8):
                filtered_categories.append(category.title())
        
        # If we don't have enough categories, add some common ones
        if len(filtered_categories) < 4:
            common_categories = ["Pizza", "Chinese", "Italian", "Japanese", "Indian", "Mexican", "Thai", "American"]
            for cat in common_categories:
                if cat not in filtered_categories and len(filtered_categories) < 8:
                    filtered_categories.append(cat)
        
        print(f"Extracted categories: {filtered_categories}")
        return filtered_categories
        
    except Exception as e:
        print(f"Error extracting categories: {e}")
        import traceback
        traceback.print_exc()
        # Fallback categories
        return ["Pizza", "Chinese", "Italian", "Japanese", "Indian", "Mexican", "Thai", "American"]

@app.route('/categories', methods=['GET'])
def get_categories():
    """Get ML-extracted categories from dataset"""
    try:
        categories = extract_categories_from_dataset()
        return jsonify({'categories': categories})
    except Exception as e:
        print(f"Error getting categories: {e}")
        # Return fallback categories
        return jsonify({'categories': ["Pizza", "Chinese", "Italian", "Japanese", "Indian", "Mexican", "Thai", "American"]})

@app.route('/categories/grouped', methods=['GET'])
def get_grouped_categories():
    """Get ML-grouped categories with actual items from dataset"""
    try:
        grouped_categories = extract_grouped_categories_from_dataset()
        return jsonify(grouped_categories)
    except Exception as e:
        print(f"Error getting grouped categories: {e}")
        # Return fallback grouped categories
        return jsonify({
            "Pizza": ["Domino's Pizza", "Pizza Hut", "Papa John's"],
            "Café": ["Starbucks", "Costa Coffee", "Dunkin' Donuts"],
            "Chinese": ["Panda Express", "PF Chang's", "China Garden"],
            "Italian": ["Olive Garden", "Macaroni Grill", "Buca di Beppo"],
            "Indian": ["Curry House", "Tandoori Palace", "Spice Garden"],
            "Mexican": ["Chipotle", "Taco Bell", "Qdoba"]
        })

@app.route('/category-items', methods=['POST'])
def get_category_items():
    """Get items for a specific category using ML similarity"""
    try:
        data = request.get_json(force=True) or {}
        category = str(data.get('category', '')).strip()
        
        if not category:
            return jsonify({'items': []})
        
        # Log category click interaction (DISABLED)
        # log_user_interaction('category_click', {'category': category})
        
        print(f"Getting items for category: {category}")
        
        # Use the existing chatbot vectorizer if available
        if chatbot_vectorizer is not None and chatbot_vectors is not None and chatbot_items:
            # Find similar restaurants using cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Transform the category query
            query_vec = chatbot_vectorizer.transform([category])
            
            # Calculate similarity scores
            scores = cosine_similarity(query_vec, chatbot_vectors)
            
            # Get top 15 most similar restaurants
            top_indices = scores[0].argsort()[::-1][:15]
            
            # Filter results with meaningful similarity (lower threshold for better results)
            items = []
            for idx in top_indices:
                if scores[0][idx] > 0.01:  # Lower threshold for better results
                    restaurant = chatbot_items[idx]
                    items.append({
                        'name': restaurant['name'],
                        'cuisine': restaurant.get('cuisine', ''),
                        'city': restaurant.get('city', ''),
                        'description': f"{restaurant['cuisine']} restaurant in {restaurant['city']}" if restaurant.get('cuisine') and restaurant.get('city') else restaurant['name']
                    })
            
            if items:
                print(f"Found {len(items)} items for category '{category}' using ML similarity")
                return jsonify({'items': items[:10]})  # Limit to 10 items
        
        # Fallback: search by keyword matching
        fallback_items = get_fallback_category_items(category)
        print(f"Using fallback search, found {len(fallback_items)} items for category '{category}'")
        return jsonify({'items': fallback_items[:10]})  # Limit to 10 items
        
    except Exception as e:
        print(f"Error getting category items: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'items': []})

def get_fallback_category_items(category):
    """Fallback method to get items by keyword matching"""
    items = []
    category_lower = category.lower()
    
    # Define keyword mappings for fallback
    keyword_mappings = {
        'pizza': ['pizza', 'pizzeria', 'domino', 'hut', 'papa'],
        'café': ['cafe', 'coffee', 'starbucks', 'costa', 'dunkin', 'brew'],
        'chinese': ['chinese', 'china', 'panda', 'wok', 'dragon', 'bamboo'],
        'italian': ['italian', 'pasta', 'pizzeria', 'trattoria', 'ristorante'],
        'indian': ['indian', 'curry', 'tandoor', 'spice', 'masala', 'biryani'],
        'mexican': ['mexican', 'taco', 'burrito', 'chipotle', 'salsa'],
        'japanese': ['japanese', 'sushi', 'ramen', 'teriyaki', 'tempura'],
        'bakery': ['bakery', 'bread', 'cake', 'pastry', 'donut', 'muffin'],
        'american': ['american', 'burger', 'fries', 'mcdonald', 'kfc', 'subway'],
        'desserts': ['dessert', 'ice cream', 'gelato', 'sweet', 'chocolate'],
        'thai': ['thai', 'thailand', 'pad thai', 'tom yum'],
        'korean': ['korean', 'korea', 'kimchi', 'bulgogi'],
        'french': ['french', 'france', 'bistro', 'brasserie'],
        'seafood': ['seafood', 'fish', 'crab', 'lobster', 'shrimp']
    }
    
    # Find matching keywords
    matching_keywords = []
    for key, keywords in keyword_mappings.items():
        if key in category_lower or any(kw in category_lower for kw in keywords):
            matching_keywords.extend(keywords)
    
    if not matching_keywords:
        matching_keywords = [category_lower]
    
    print(f"Searching for keywords: {matching_keywords}")
    
    # Search restaurants for matching keywords
    for restaurant in restaurants:
        # Use cleaned dataset format
        name = restaurant.get("Restaurant_Name", "")
        cuisine = restaurant.get("Cuisines", "")
        city = restaurant.get("City", "")
        
        if name:
            # Check if any keyword matches
            text_to_search = f"{name} {cuisine} {city}".lower()
            if any(keyword in text_to_search for keyword in matching_keywords):
                items.append({
                    'name': name,
                    'cuisine': cuisine,
                    'city': city,
                    'description': f"{cuisine} restaurant in {city}" if cuisine and city else name
                })
                
                if len(items) >= 15:  # Limit to 15 items for better selection
                    break
    
    print(f"Found {len(items)} fallback items for category '{category}'")
    return items

def extract_grouped_categories_from_dataset():
    """Extract and group items from restaurant dataset using ML clustering"""
    try:
        print("Extracting grouped categories from restaurant dataset...")
        
        # Prepare restaurant data for clustering
        restaurant_data = []
        for restaurant in restaurants:
            name = restaurant.get("Restaurant_Name", "")
            cuisine = restaurant.get("Cuisines", "")
            city = restaurant.get("City", "")
            
            if name:  # Only include restaurants with names
                # Create combined text for clustering
                combined_text = f"{name} {cuisine} {city}".strip()
                restaurant_data.append({
                    'name': name,
                    'cuisine': cuisine,
                    'city': city,
                    'combined_text': combined_text
                })
        
        if len(restaurant_data) < 6:
            print("Not enough data for clustering, using fallback grouping")
            return create_fallback_grouped_categories(restaurant_data)
        
        # Use TF-IDF vectorization for text clustering
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        
        # Prepare text data for clustering
        texts = [item['combined_text'] for item in restaurant_data]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1
        )
        X = vectorizer.fit_transform(texts)
        
        # Determine optimal number of clusters (between 4 and 8)
        n_clusters = min(8, max(4, len(restaurant_data) // 10))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Assign cluster labels to restaurants
        for i, item in enumerate(restaurant_data):
            item['cluster'] = cluster_labels[i]
        
        # Group restaurants by cluster
        clusters = {}
        for item in restaurant_data:
            cluster_id = item['cluster']
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(item)
        
        # Map clusters to meaningful category names
        category_mapping = map_clusters_to_categories(clusters, vectorizer)
        
        # Create final grouped categories
        grouped_categories = {}
        for cluster_id, items in clusters.items():
            category_name = category_mapping.get(cluster_id, f"Category {cluster_id}")
            restaurant_names = [item['name'] for item in items[:10]]  # Limit to 10 items per category
            grouped_categories[category_name] = restaurant_names
        
        print(f"Created {len(grouped_categories)} grouped categories")
        return grouped_categories
        
    except Exception as e:
        print(f"Error in grouped category extraction: {e}")
        return create_fallback_grouped_categories([])

def map_clusters_to_categories(clusters, vectorizer):
    """Map cluster IDs to meaningful category names based on content analysis"""
    category_mapping = {}
    
    # Define keyword patterns for different categories
    category_keywords = {
        'Pizza': ['pizza', 'pizzeria', 'domino', 'hut', 'papa'],
        'Café': ['cafe', 'coffee', 'starbucks', 'costa', 'dunkin', 'brew'],
        'Chinese': ['chinese', 'china', 'panda', 'wok', 'dragon', 'bamboo'],
        'Italian': ['italian', 'pasta', 'pizzeria', 'trattoria', 'ristorante'],
        'Indian': ['indian', 'curry', 'tandoor', 'spice', 'masala', 'biryani'],
        'Mexican': ['mexican', 'taco', 'burrito', 'chipotle', 'salsa'],
        'Japanese': ['japanese', 'sushi', 'ramen', 'teriyaki', 'tempura'],
        'Bakery': ['bakery', 'bread', 'cake', 'pastry', 'donut', 'muffin'],
        'Fast Food': ['burger', 'fries', 'mcdonald', 'kfc', 'subway'],
        'Desserts': ['dessert', 'ice cream', 'gelato', 'sweet', 'chocolate']
    }
    
    for cluster_id, items in clusters.items():
        # Analyze the most common words in this cluster
        all_text = ' '.join([item['combined_text'].lower() for item in items])
        
        # Find the best matching category
        best_match = None
        best_score = 0
        
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            if score > best_score:
                best_score = score
                best_match = category
        
        if best_match and best_score > 0:
            category_mapping[cluster_id] = best_match
        else:
            # Fallback: use the most common cuisine in the cluster
            cuisines = [item['cuisine'] for item in items if item['cuisine']]
            if cuisines:
                from collections import Counter
                most_common_cuisine = Counter(cuisines).most_common(1)[0][0]
                category_mapping[cluster_id] = most_common_cuisine.title()
            else:
                category_mapping[cluster_id] = f"Category {cluster_id + 1}"
    
    return category_mapping

def create_fallback_grouped_categories(restaurant_data):
    """Create fallback grouped categories when clustering fails"""
    # Group by cuisine as fallback
    cuisine_groups = {}
    for item in restaurant_data:
        cuisine = item.get('cuisine', 'Other')
        if cuisine not in cuisine_groups:
            cuisine_groups[cuisine] = []
        cuisine_groups[cuisine].append(item['name'])
    
    # Convert to the expected format
    grouped = {}
    for cuisine, names in cuisine_groups.items():
        if names:
            grouped[cuisine.title()] = names[:8]  # Limit to 8 items per category
    
    return grouped


# ---------- API: Trending Recommendations ----------
@app.route("/api/recommendations")
def trending_recommendations():
    # sort by rating then votes (defensive conversions)
    def key_func(r):
        try:
            return (float(r.get("Rating", 0) or 0), int(float(r.get("Votes", 0) or 0)))
        except Exception:
            return (0, 0)

    trending = sorted(restaurants, key=key_func, reverse=True)
    return jsonify(trending[:10])


# ---------- API: ML-based Recommendations ----------
@app.route("/api/recommend")
def get_recommendations():
    name = request.args.get("name", "")
    results = recommend_restaurants(name)
    return jsonify(results)

# ---------- Advanced ML Recommendation Endpoint ----------
@app.route("/recommend", methods=["POST"])
@cache.memoize(timeout=300)  # Cache for 5 minutes
def get_ml_recommendations():
    """
    Advanced ML recommendation endpoint that takes:
    {
        "user_input": "North Indian food",
        "mood": "happy",
        "time": "evening", 
        "occasion": "dinner"
    }
    """
    try:
        if not ML_ENGINE_AVAILABLE:
            return jsonify({
                "error": "ML recommendation engine not available",
                "recommendations": []
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "recommendations": []
            }), 400
        
        # Extract parameters
        user_input = data.get("user_input", "").strip()
        mood = data.get("mood", "happy").strip().lower()
        time = data.get("time", "evening").strip().lower()
        occasion = data.get("occasion", "dinner").strip().lower()
        
        # Validate parameters
        valid_moods = ["happy", "sad", "angry", "relaxed", "excited", "bored"]
        valid_times = ["morning", "afternoon", "evening", "night"]
        valid_occasions = ["birthday", "date", "party", "lunch", "dinner", "meeting", "anniversary"]
        
        if mood not in valid_moods:
            mood = "happy"
        if time not in valid_times:
            time = "evening"
        if occasion not in valid_occasions:
            occasion = "dinner"
        
        if not user_input:
            user_input = "restaurant"
        
        # Log the recommendation request (DISABLED)
        # log_user_interaction('ml_recommendation', {
        #     'user_input': user_input,
        #     'mood': mood,
        #     'time': time,
        #     'occasion': occasion
        # })
        
        # Get recommendations from ML engine
        recommendations = recommend_restaurants(user_input, n=5)
        
        # Format recommendations for response
        formatted_recommendations = []
        for rec in recommendations:
            formatted_recommendations.append({
                "name": rec.get("name", ""),
                "cuisines": rec.get("cuisines", ""),
                "rating": rec.get("rating", 0),
                "address": rec.get("location", ""),
                "cost": rec.get("cost", 0),
                "match_score": round(rec.get("match_score", 0), 3),
                "type": rec.get("type", "hybrid")
            })
        
        return jsonify({
            "recommendations": formatted_recommendations,
            "query": {
                "user_input": user_input,
                "mood": mood,
                "time": time,
                "occasion": occasion
            },
            "total": len(formatted_recommendations)
        })
        
    except Exception as e:
        print(f"Error in ML recommendations: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "recommendations": []
        }), 500

# ---------- Model Loading Functions ----------
def load_trained_models():
    """Load the trained models from the models directory"""
    try:
        import pickle
        model_path = os.path.join("models", "model.pkl")
        similarity_path = os.path.join("models", "similarity.pkl")
        
        if os.path.exists(model_path) and os.path.exists(similarity_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            with open(similarity_path, 'rb') as f:
                similarity_matrix = pickle.load(f)
            
            print(f"Loaded trained models: {len(model_data['restaurant_data'])} restaurants")
            return model_data, similarity_matrix
        else:
            print("Trained models not found, using basic similarity")
            return None, None
    except Exception as e:
        print(f"Error loading trained models: {e}")
        return None, None

# Load trained models if available
trained_model_data, trained_similarity = load_trained_models()

# ---------- ML Engine Status Endpoint ----------
@app.route("/ml/status", methods=["GET"])
def ml_engine_status():
    """Check ML engine status and availability"""
    try:
        if not ML_ENGINE_AVAILABLE:
            return jsonify({
                "available": False,
                "message": "ML recommendation engine not available"
            })
        
        # Check if models are loaded
        models_loaded = (
            recommendation_engine.tfidf_vectorizer is not None and
            recommendation_engine.restaurants is not None
        )
        
        return jsonify({
            "available": True,
            "models_loaded": models_loaded,
            "restaurant_count": len(recommendation_engine.restaurants) if recommendation_engine.restaurants else 0,
            "content_based": recommendation_engine.tfidf_vectorizer is not None,
            "collaborative": recommendation_engine.lightfm_model is not None,
            "context_aware": recommendation_engine.sentence_model is not None,
            "trained_models_available": trained_model_data is not None
        })
        
    except Exception as e:
        return jsonify({
            "available": False,
            "error": str(e)
        }), 500


# ---------- API: Categories ----------
@app.route("/api/categories", methods=["GET"])
def api_categories():
    global categories_cache
    if categories_cache is None:
        compute_categories()
    return jsonify(categories_cache or {})

# ---------- Enhanced API: Search and Filtering ----------
@app.route("/api/search", methods=["GET", "POST"])
def search_restaurants():
    """Enhanced search with filters and pagination"""
    try:
        if request.method == "POST":
            data = request.json or {}
        else:
            data = request.args.to_dict()
        
        # Extract search parameters
        query = data.get("q", "").strip()
        city = data.get("city", "").strip()
        cuisine = data.get("cuisine", "").strip()
        rating_min = float(data.get("rating_min", 0))
        price_max = float(data.get("price_max", float('inf')))
        mood = data.get("mood", "").strip()
        time_of_day = data.get("time", "").strip()
        occasion = data.get("occasion", "").strip()
        page = int(data.get("page", 1))
        per_page = int(data.get("per_page", 20))
        sort_by = data.get("sort", "rating")  # rating, price, distance, name
        
        # Start with all restaurants - use global restaurants list
        global restaurants
        if not restaurants or len(restaurants) == 0:
            print("Debug: No restaurants data available")
            return jsonify({
                "restaurants": [],
                "total": 0,
                "page": page,
                "per_page": per_page,
                "total_pages": 0
            })
        
        # Convert to DataFrame for easier filtering
        results = pd.DataFrame(restaurants)
        print(f"Debug: Starting with {len(results)} restaurants")
        
        # Apply filters
        if query:
            # Text search in restaurant name, cuisines, and city
            search_text = (results["Restaurant Name"].fillna("").astype(str) + " " + 
                          results["Cuisines"].fillna("").astype(str) + " " + 
                          results["City"].fillna("").astype(str)).str.lower()
            mask = search_text.str.contains(query.lower(), na=False)
            results = results[mask]
        
        if city:
            mask = results["City"].fillna("").str.contains(city, case=False, na=False)
            results = results[mask]
        
        if cuisine:
            mask = results["Cuisines"].fillna("").str.contains(cuisine, case=False, na=False)
            results = results[mask]
        
        if rating_min > 0:
            results = results[results["Aggregate rating"] >= rating_min]
        
        if price_max < float('inf'):
            results = results[results["Average Cost for two"] <= price_max]
        
        # Apply mood/time/occasion filters using ML logic
        if mood or time_of_day or occasion:
            results = apply_contextual_filters(results, mood, time_of_day, occasion)
        
        # Sort results
        if sort_by == "rating":
            results = results.sort_values("Aggregate rating", ascending=False)
        elif sort_by == "price":
            results = results.sort_values("Average Cost for two", ascending=True)
        elif sort_by == "name":
            results = results.sort_values("Restaurant Name", ascending=True)
        
        # Pagination
        total = len(results)
        total_pages = (total + per_page - 1) // per_page
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_results = results.iloc[start_idx:end_idx]
        
        # Format results
        restaurants = []
        for _, restaurant in page_results.iterrows():
            restaurants.append({
                "id": restaurant.get("Restaurant Name", ""),
                "name": restaurant.get("Restaurant Name", ""),
                "cuisines": restaurant.get("Cuisines", ""),
                "city": restaurant.get("City", ""),
                "location": restaurant.get("Location", ""),
                "address": restaurant.get("Address", ""),
                "rating": float(restaurant.get("Aggregate rating", 0)),
                "votes": int(restaurant.get("Votes", 0)),
                "price": float(restaurant.get("Average Cost for two", 0)),
                "price_range": f"₹{restaurant.get('Average Cost for two', 0)}",
                "type": restaurant.get("Restaurant Type", ""),
                "latitude": float(restaurant.get("Latitude", 0)) if pd.notna(restaurant.get("Latitude")) else None,
                "longitude": float(restaurant.get("Longitude", 0)) if pd.notna(restaurant.get("Longitude")) else None,
                "online_order": restaurant.get("Online Order", "No"),
                "book_table": restaurant.get("Book Table", "No")
            })
        
        return jsonify({
            "restaurants": restaurants,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "filters_applied": {
                "query": query,
                "city": city,
                "cuisine": cuisine,
                "rating_min": rating_min,
                "price_max": price_max,
                "mood": mood,
                "time": time_of_day,
                "occasion": occasion
            }
        })
        
    except Exception as e:
        print(f"Error in search_restaurants: {e}")
        return jsonify({"error": "Search failed", "details": str(e)}), 500

def apply_contextual_filters(df, mood, time_of_day, occasion):
    """Apply mood, time, and occasion filters using ML logic"""
    if df.empty:
        return df
    
    # Rule-based mappings for mood/time/occasion
    mood_cuisines = {
        "happy": ["Desserts", "Ice Cream", "Cafe", "Bakery", "Fast Food", "American", "Pizza", "Burger", "Finger Food", "Beverages"],
        "sad": ["Chinese", "Italian", "Desserts", "North Indian", "South Indian", "Comfort Food", "Soups", "Noodles"],
        "angry": ["North Indian", "South Indian", "Chinese", "Thai", "Fast Food", "Spicy", "Curry", "Biryani"],
        "relaxed": ["Cafe", "Coffee", "Mediterranean", "Continental", "Bakery", "Tea", "Light Bites", "Salads"],
        "excited": ["Fast Food", "Pizza", "Burger", "American", "Continental", "Finger Food", "Snacks", "Street Food"],
        "bored": ["Continental", "Chinese", "Italian", "North Indian", "South Indian", "Fusion", "International", "Multi-cuisine"]
    }
    
    time_cuisines = {
        "morning": ["Cafe", "Coffee", "Bakery", "Continental", "Quick Bites", "Breakfast", "Tea", "Light Bites"],
        "afternoon": ["North Indian", "South Indian", "Chinese", "Continental", "Quick Bites", "Lunch", "Thali", "Combo"],
        "evening": ["North Indian", "South Indian", "Chinese", "Continental", "Italian", "Dinner", "Fine Dining", "Multi-cuisine"],
        "night": ["Fast Food", "Pizza", "Burger", "North Indian", "South Indian", "Late Night", "Street Food", "Snacks"]
    }
    
    occasion_cuisines = {
        "date": ["Italian", "French", "Mediterranean", "Continental", "Fine Dining", "Romantic", "Candle Light"],
        "birthday": ["Desserts", "Cake", "American", "Continental", "Finger Food", "Party", "Celebration"],
        "party": ["Fast Food", "Pizza", "Burger", "Finger Food", "American", "Snacks", "Street Food"],
        "meeting": ["Cafe", "Coffee", "Continental", "Quick Bites", "Bakery", "Business", "Professional"],
        "anniversary": ["Italian", "French", "Mediterranean", "Continental", "Fine Dining", "Romantic", "Special"],
        "lunch": ["North Indian", "South Indian", "Chinese", "Continental", "Quick Bites", "Thali", "Combo"],
        "dinner": ["North Indian", "South Indian", "Chinese", "Continental", "Italian", "Fine Dining", "Multi-cuisine"]
    }
    
    # Apply filters
    mask = pd.Series([True] * len(df), index=df.index)
    
    if mood and mood in mood_cuisines:
        mood_mask = df["Cuisines"].fillna("").str.contains("|".join(mood_cuisines[mood]), case=False, na=False)
        mask = mask & mood_mask
    
    if time_of_day and time_of_day in time_cuisines:
        time_mask = df["Cuisines"].fillna("").str.contains("|".join(time_cuisines[time_of_day]), case=False, na=False)
        mask = mask & time_mask
    
    if occasion and occasion in occasion_cuisines:
        occasion_mask = df["Cuisines"].fillna("").str.contains("|".join(occasion_cuisines[occasion]), case=False, na=False)
        mask = mask & occasion_mask
    
    return df[mask]

@app.route("/api/restaurant/<restaurant_name>")
def get_restaurant_details(restaurant_name):
    """Get detailed information about a specific restaurant"""
    try:
        if df is None or df.empty:
            return jsonify({"error": "Restaurant data not available"}), 404
        
        # Find restaurant by name
        restaurant = df[df["Restaurant Name"] == restaurant_name]
        if restaurant.empty:
            return jsonify({"error": "Restaurant not found"}), 404
        
        restaurant = restaurant.iloc[0]
        
        # Get similar restaurants
        similar = recommend_restaurants(restaurant_name, n=5)
        
        details = {
            "id": restaurant.get("Restaurant Name", ""),
            "name": restaurant.get("Restaurant Name", ""),
            "cuisines": restaurant.get("Cuisines", ""),
            "city": restaurant.get("City", ""),
            "location": restaurant.get("Location", ""),
            "address": restaurant.get("Address", ""),
            "rating": float(restaurant.get("Aggregate rating", 0)),
            "votes": int(restaurant.get("Votes", 0)),
            "price": float(restaurant.get("Average Cost for two", 0)),
            "price_range": restaurant.get("Price_Range", "₹0"),
            "type": restaurant.get("Restaurant Type", ""),
            "latitude": float(restaurant.get("Latitude", 0)),
            "longitude": float(restaurant.get("Longitude", 0)),
            "online_order": restaurant.get("Online Order", "No"),
            "book_table": restaurant.get("Book Table", "No"),
            "dish_liked": restaurant.get("Dish Liked", ""),
            "reviews_list": restaurant.get("Reviews List", ""),
            "cuisine_list": restaurant.get("Cuisine List", ""),
            "price_category": restaurant.get("Price Category", ""),
            "rating_category": restaurant.get("Rating Category", ""),
            "similar_restaurants": similar
        }
        
        return jsonify(details)
        
    except Exception as e:
        print(f"Error in get_restaurant_details: {e}")
        return jsonify({"error": "Failed to get restaurant details", "details": str(e)}), 500

def get_similar_restaurants(restaurant_name, limit=6):
    """Get similar restaurants for the details page"""
    try:
        # Use the existing recommendation function
        similar = recommend_restaurants(restaurant_name, n=limit)
        return similar
    except Exception as e:
        print(f"Error in get_similar_restaurants: {e}")
        return []

@app.route("/api/nearby", methods=["GET", "POST"])
def get_nearby_restaurants():
    """Get restaurants near a location with distance calculations"""
    try:
        if request.method == "POST":
            data = request.json or {}
        else:
            data = request.args.to_dict()
        
        # Get location parameters
        latitude = float(data.get("lat", 0))
        longitude = float(data.get("lng", 0))
        city = data.get("city", "").strip()
        radius_km = float(data.get("radius", 10))  # Default 10km radius
        
        if df is None or df.empty:
            return jsonify({"restaurants": [], "total": 0})
        
        results = df.copy()
        
        # If coordinates provided, calculate distances
        if latitude != 0 and longitude != 0:
            results = calculate_distances(results, latitude, longitude)
            results = results[results["distance_km"] <= radius_km]
            results = results.sort_values("distance_km")
        elif city:
            # Fallback to city-based filtering
            results = results[results["City"].str.contains(city, case=False, na=False)]
        
        # Format results
        restaurants = []
        for _, restaurant in results.iterrows():
            distance_text = ""
            if latitude != 0 and longitude != 0 and "distance_km" in restaurant:
                distance = restaurant["distance_km"]
                if distance < 1:
                    distance_text = f"{distance*1000:.0f}m"
                else:
                    distance_text = f"{distance:.1f}km"
            
            restaurants.append({
                "id": restaurant.get("Restaurant_Name", ""),
                "name": restaurant.get("Restaurant_Name", ""),
                "cuisines": restaurant.get("Cuisines", ""),
                "city": restaurant.get("City", ""),
                "location": restaurant.get("Location", ""),
                "address": restaurant.get("Address", ""),
                "rating": float(restaurant.get("Rating", 0)),
                "votes": int(restaurant.get("Votes", 0)),
                "price": float(restaurant.get("Price", 0)),
                "price_range": restaurant.get("Price_Range", "₹0"),
                "type": restaurant.get("Restaurant_Type", ""),
                "latitude": float(restaurant.get("Latitude", 0)) if pd.notna(restaurant.get("Latitude")) else None,
                "longitude": float(restaurant.get("Longitude", 0)) if pd.notna(restaurant.get("Longitude")) else None,
                "distance": distance_text,
                "distance_km": float(restaurant.get("distance_km", 0)) if "distance_km" in restaurant else 0
            })
        
        return jsonify({
            "restaurants": restaurants,
            "total": len(restaurants),
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "city": city,
                "radius_km": radius_km
            }
        })
        
    except Exception as e:
        print(f"Error in get_nearby_restaurants: {e}")
        return jsonify({"error": "Failed to get nearby restaurants", "details": str(e)}), 500

def calculate_distances(df, user_lat, user_lng):
    """Calculate distances from user location to restaurants"""
    from math import radians, cos, sin, asin, sqrt
    
    def haversine(lon1, lat1, lon2, lat2):
        """Calculate the great circle distance between two points on earth"""
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r
    
    # Calculate distances
    distances = []
    for _, restaurant in df.iterrows():
        rest_lat = restaurant.get("Latitude", 0)
        rest_lng = restaurant.get("Longitude", 0)
        
        if rest_lat != 0 and rest_lng != 0:
            distance = haversine(user_lng, user_lat, rest_lng, rest_lat)
        else:
            distance = float('inf')  # No coordinates available
        
        distances.append(distance)
    
    df["distance_km"] = distances
    return df

@app.route("/api/debug", methods=["GET"])
def debug_endpoint():
    """Debug endpoint to check DataFrame status"""
    global df, restaurants
    return jsonify({
        "restaurants_length": len(restaurants) if restaurants else 0,
        "restaurants_sample": restaurants[:2] if restaurants and len(restaurants) > 0 else None,
        "df_is_none": df is None,
        "df_empty": df.empty if df is not None else True,
        "df_shape": df.shape if df is not None else None,
        "df_columns": list(df.columns) if df is not None else None,
        "sample_combined_features": df["Combined_Features"].head().tolist() if df is not None and "Combined_Features" in df.columns else None
    })

@app.route("/api/featured", methods=["GET"])
def get_featured_restaurants():
    """Get featured restaurants based on rating, popularity, and recency"""
    try:
        global restaurants
        if not restaurants or len(restaurants) == 0:
            return jsonify({"restaurants": [], "total": 0})
        
        # Convert to DataFrame for easier processing
        df_featured = pd.DataFrame(restaurants)
        
        # Featured selection logic: high rating + high votes + recent
        featured = df_featured.copy()
        
        # Filter for high-quality restaurants using correct column names
        # Convert to numeric and handle any missing values
        featured["Aggregate rating"] = pd.to_numeric(featured["Aggregate rating"], errors='coerce').fillna(0)
        featured["Votes"] = pd.to_numeric(featured["Votes"], errors='coerce').fillna(0)
        featured["Average Cost for two"] = pd.to_numeric(featured["Average Cost for two"], errors='coerce').fillna(0)
        
        # Filter for high-quality restaurants
        featured = featured[featured["Aggregate rating"] >= 4.0]
        featured = featured[featured["Votes"] >= 100]
        
        # If no restaurants meet strict criteria, relax the requirements
        if len(featured) < 5:
            print("Relaxing featured restaurant criteria...")
            featured = df_featured.copy()
            featured["Aggregate rating"] = pd.to_numeric(featured["Aggregate rating"], errors='coerce').fillna(0)
            featured["Votes"] = pd.to_numeric(featured["Votes"], errors='coerce').fillna(0)
            featured["Average Cost for two"] = pd.to_numeric(featured["Average Cost for two"], errors='coerce').fillna(0)
            
            # More relaxed criteria
            featured = featured[featured["Aggregate rating"] >= 3.5]
            featured = featured[featured["Votes"] >= 50]
        
        # Calculate featured score (weighted combination)
        featured["featured_score"] = (
            featured["Aggregate rating"] * 0.4 +
            (featured["Votes"] / featured["Votes"].max()) * 0.3 +
            (1 - featured["Average Cost for two"] / featured["Average Cost for two"].max()) * 0.3  # Lower price = higher score
        )
        
        # Sort by featured score and take top 12
        featured = featured.sort_values("featured_score", ascending=False).head(12)
        
        # Format results
        restaurants_list = []
        for _, restaurant in featured.iterrows():
            restaurants_list.append({
                "id": restaurant.get("Restaurant Name", ""),
                "name": restaurant.get("Restaurant Name", ""),
                "cuisines": restaurant.get("Cuisines", ""),
                "city": restaurant.get("City", ""),
                "location": restaurant.get("Location", ""),
                "address": restaurant.get("Address", ""),
                "rating": float(restaurant.get("Aggregate rating", 0)),
                "votes": int(restaurant.get("Votes", 0)),
                "price": float(restaurant.get("Average Cost for two", 0)),
                "price_range": f"₹{int(restaurant.get('Average Cost for two', 0))}",
                "type": restaurant.get("Restaurant Type", ""),
                "latitude": float(restaurant.get("Latitude", 0)) if pd.notna(restaurant.get("Latitude")) else None,
                "longitude": float(restaurant.get("Longitude", 0)) if pd.notna(restaurant.get("Longitude")) else None,
                "online_order": restaurant.get("Online Order", "No"),
                "book_table": restaurant.get("Book Table", "No"),
                "dish_liked": restaurant.get("Dish Liked", ""),
                "featured_score": float(restaurant.get("featured_score", 0))
            })
        
        return jsonify({
            "restaurants": restaurants_list,
            "total": len(restaurants_list),
            "selection_criteria": "High rating (4.0+), high votes (100+), balanced price-quality ratio"
        })
        
    except Exception as e:
        print(f"Error in get_featured_restaurants: {e}")
        return jsonify({"error": "Failed to get featured restaurants", "details": str(e)}), 500

# ---------- API: Wishlist ----------
@app.route("/api/wishlist", methods=["GET"])
def get_wishlist():
    return jsonify(load_wishlist())


@app.route("/api/wishlist", methods=["POST"])
def add_to_wishlist():
    data = request.json or {}
    wishlist = load_wishlist()
    name = data.get("name") or data.get("Restaurant_Name") or ""
    if not name:
        return jsonify({"message": "Missing name"}), 400

    username = session.get("user") or ""
    for item in wishlist:
        iname = item.get("name") or item.get("Restaurant_Name")
        iuser = item.get("username") or ""
        if iname == name and iuser == username:
            return jsonify({"message": "Already in wishlist"}), 400

    wishlist.append({"name": name, "username": username})
    save_wishlist(wishlist)
    return jsonify({"message": "Added to wishlist"}), 201


@app.route("/api/wishlist/<name>", methods=["DELETE"])
def remove_from_wishlist(name):
    wishlist = load_wishlist()
    username = session.get("user") or ""
    new_list = [item for item in wishlist if not (((item.get("name") or item.get("Restaurant_Name")) == name) and ((item.get("username") or "") == username))]
    save_wishlist(new_list)
    return jsonify({"message": "Removed from wishlist"})


# Form-based removal for wishlist page
@app.route("/wishlist/remove", methods=["POST"])
def remove_wishlist_item():
    name = request.form.get("item_id", "").strip()
    if not name:
        return redirect(url_for("wishlist_page"))
    wishlist = load_wishlist()
    username = session.get("user") or ""
    new_list = [item for item in wishlist if not (((item.get("name") or item.get("Restaurant_Name")) == name) and ((item.get("username") or "") == username))]
    save_wishlist(new_list)
    return redirect(url_for("wishlist_page"))


# Detailed wishlist for current user (full restaurant objects)
@app.route("/api/wishlist/details", methods=["GET"])
def wishlist_details():
    username = session.get("user") or ""
    wl = load_wishlist()
    names = { (item.get("name") or item.get("Restaurant_Name")) for item in wl if (item.get("username") or "") == username }
    if not names:
        return jsonify([])
    detailed = [r for r in restaurants if (r.get("Restaurant_Name") or "") in names]
    return jsonify(detailed)


# ---------- Admin Panel ----------

# User interaction tracking
USER_INTERACTIONS_FILE = 'data/user_interactions.json'

def log_user_interaction(interaction_type, data):
    """Log user interactions for admin analytics"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Load existing interactions
        interactions = []
        if os.path.exists(USER_INTERACTIONS_FILE):
            try:
                with open(USER_INTERACTIONS_FILE, 'r') as f:
                    interactions = json.load(f)
            except:
                interactions = []
        
        # Add new interaction
        interaction = {
            'timestamp': datetime.datetime.now().isoformat(),
            'type': interaction_type,
            'data': data,
            'ip': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', '')
        }
        interactions.append(interaction)
        
        # Keep only last 1000 interactions to prevent file from growing too large
        if len(interactions) > 1000:
            interactions = interactions[-1000:]
        
        # Save to file
        with open(USER_INTERACTIONS_FILE, 'w') as f:
            json.dump(interactions, f, indent=2)
            
    except Exception as e:
        print(f"Error logging user interaction: {e}")

def load_user_interactions():
    """Load user interactions for admin panel"""
    try:
        if os.path.exists(USER_INTERACTIONS_FILE):
            with open(USER_INTERACTIONS_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return []

@app.route('/admin')
def admin_login():
    """Admin login page"""
    if session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard'))
    return render_template('admin_login.html')

@app.route('/admin/login', methods=['POST'])
def admin_login_post():
    """Handle admin login"""
    username = request.form.get('username')
    password = request.form.get('password')
    
    if username == 'admin' and password == 'admin':
        session['admin_logged_in'] = True
        return redirect(url_for('admin_dashboard'))
    else:
        flash('Invalid credentials', 'error')
        return redirect(url_for('admin_login'))

@app.route('/admin/dashboard')
def admin_dashboard():
    """Admin dashboard with analytics"""
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    # Load user interactions
    interactions = load_user_interactions()
    
    # Analyze interactions
    search_queries = []
    category_clicks = []
    chatbot_queries = []
    hourly_usage = defaultdict(int)
    
    for interaction in interactions:
        # Parse timestamp for hourly analysis
        try:
            timestamp = datetime.datetime.fromisoformat(interaction['timestamp'])
            hour = timestamp.hour
            hourly_usage[hour] += 1
        except:
            pass
        
        # Categorize interactions
        if interaction['type'] == 'search':
            search_queries.append(interaction['data'].get('query', ''))
        elif interaction['type'] == 'category_click':
            category_clicks.append(interaction['data'].get('category', ''))
        elif interaction['type'] == 'chatbot':
            chatbot_queries.append(interaction['data'].get('query', ''))
    
    # Get most popular items
    most_searched = Counter(search_queries).most_common(10)
    most_clicked_categories = Counter(category_clicks).most_common(10)
    most_chatbot_queries = Counter(chatbot_queries).most_common(10)
    
    # Get recent interactions (last 50)
    recent_interactions = interactions[-50:] if interactions else []
    
    # Calculate stats
    total_interactions = len(interactions)
    unique_ips = len(set(interaction['ip'] for interaction in interactions))
    
    return render_template('admin_dashboard.html',
                         total_interactions=total_interactions,
                         unique_ips=unique_ips,
                         most_searched=most_searched,
                         most_clicked_categories=most_clicked_categories,
                         most_chatbot_queries=most_chatbot_queries,
                         recent_interactions=recent_interactions,
                         hourly_usage=dict(hourly_usage))

@app.route('/admin/logout')
def admin_logout():
    """Admin logout"""
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))


# ---------- Run ----------
if __name__ == "__main__":
    # Initialize ML recommendation engine
    if ML_ENGINE_AVAILABLE:
        def _init_ml_engine():
            try:
                print("Initializing ML recommendation engine...")
                initialize_recommendation_engine()
                print("ML recommendation engine initialized successfully!")
            except Exception as e:
                print(f"Error initializing ML recommendation engine: {e}")
                import traceback
                traceback.print_exc()
        
        # Initialize ML engine in background thread
        threading.Thread(target=_init_ml_engine, daemon=True).start()
    
    # Load persisted predictor first (non-blocking startup if available)
    try:
        import joblib
        PRED_PATH = os.path.join(os.path.dirname(__file__), '..', 'restaurant_predictor.joblib')
        if os.path.exists(PRED_PATH):
            data = joblib.load(PRED_PATH)
            if isinstance(data, dict) and 'vectorizer' in data and 'model' in data:
                suggest_vectorizer = data['vectorizer']
                suggest_model = data['model']
                print("Loaded persisted predictive typing model.")
    except Exception:
        pass

    # If no persisted model, initialize training in a background daemon thread to avoid blocking startup
    if (suggest_vectorizer is None or suggest_model is None) and _suggest_texts_for_fit and _suggest_labels:
        def _train_predictive_model():
            global suggest_vectorizer, suggest_model
            try:
                print(f"Training predictive typing model with {len(_suggest_texts_for_fit)} samples...")
                print(f"Sample data: {_suggest_texts_for_fit[:5]}")
                
                # Character-level n-grams (2,3) per requirement
                local_vectorizer = CountVectorizer(ngram_range=(2, 3), analyzer='char', min_df=1)
                Xc_local = local_vectorizer.fit_transform(_suggest_texts_for_fit)
                print(f"Vectorized features shape: {Xc_local.shape}")
                
                local_model = LogisticRegression(max_iter=300, random_state=42)
                local_model.fit(Xc_local, _suggest_labels)
                
                # assign to globals after successful fit
                suggest_vectorizer = local_vectorizer
                suggest_model = local_model
                
                # Test the model
                test_input = "piz"
                test_vec = local_vectorizer.transform([test_input])
                test_pred = local_model.predict(test_vec)
                print(f"Test prediction for 'piz': {test_pred[0]}")
                
                # Persist for next runs
                try:
                    import joblib
                    PRED_PATH = os.path.join(os.path.dirname(__file__), '..', 'restaurant_predictor.joblib')
                    joblib.dump({'vectorizer': suggest_vectorizer, 'model': suggest_model}, PRED_PATH)
                    print(f"Model saved to {PRED_PATH}")
                except Exception as e:
                    print(f"Failed to save model: {e}")
                
                print("Predictive typing model training complete.")
            except Exception as e:
                print(f"Model training failed: {e}")
                # leave as None if training fails
                suggest_vectorizer = None
                suggest_model = None

        threading.Thread(target=_train_predictive_model, daemon=True).start()

    # Initialize chatbot after data is loaded
    print("Initializing chatbot with loaded data...")
    initialize_chatbot()

    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask app on port {port}...")
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)
