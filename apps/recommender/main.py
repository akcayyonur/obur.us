from functools import lru_cache
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Set, List, Dict
import math
import pandas as pd
import os
from nlp_processor import analyze_sentiment, get_sentiment_label
from graph_candidates import get_candidates
from neo4j import GraphDatabase

# -------------------------------------------------
# App
# -------------------------------------------------

app = FastAPI(
    title="OBURUS Graph-Based Personalized Recommender",
    version="3.0"
)

# Static UI
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")

# -------------------------------------------------
# Neo4j Connection (READ-ONLY)
# -------------------------------------------------

# Build absolute paths to data files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE_PATH = os.path.join(BASE_DIR, "data", "synthetic_users.xlsx")

NEO4J_URI = "bolt://neo4j:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "oburus_pass"

neo_driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS)
)

# -------------------------------------------------
# Exception Handlers (NEW)
# -------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    import traceback
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error_message": str(exc), "traceback": traceback.format_exc().splitlines()},
    )


# -------------------------------------------------
# Preference & Scoring Constants
# -------------------------------------------------

# Mapping specific food/drink items to broader categories
FOOD_TO_CATEGORY = {
    # Turkish
    "kebap": "turkish",
    "pide": "turkish",
    "döner": "turkish",
    "lahmacun": "turkish",
    "baklava": "dessert",
    "köfte": "turkish",
    "kofte": "turkish",
    "tatlı": "dessert",
    "tatli": "dessert",
    # Italian
    "pizza": "italian",
    "pasta": "italian",
    "lasagna": "italian",
    # Cafe
    "cafe": "coffee",
    "coffee": "coffee",
    "kahve": "coffee",
    "tea": "coffee",
    "çay": "coffee",
    "espresso": "coffee",
    "latte": "coffee",
    "cappuccino": "coffee",
    "americano": "coffee",
    "nargile": "hookah",
    # Fast Food / Other
    "burger": "fast-food",
    "hamburger": "fast-food",
    "sushi": "japanese",
    "seafood": "seafood",
    "balık": "seafood",
    "balik": "seafood",
    "et": "steakhouse",
    "meat": "steakhouse",
    "steak": "steakhouse",
}

# Hybrid scoring weights
W_RATING = 0.28
W_POP    = 0.05
W_CAT    = 0.14
W_DIST   = 0.10
W_PRICE  = 0.10
W_NAME   = 0.15  # New: Score for matching preference in name
W_REVIEW = 0.25  # Increased from 0.20
W_COLLAB = 0.10
W_PAGERANK = 0.18
W_SENTIMENT = 0.10 # New: Weight for sentiment score

MAX_KM = 20.0

# -------------------------------------------------
# Helper functions (Scoring)
# -------------------------------------------------

def safe_float(x, default=0.0):
    if x is None:
        return default
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return default
    return float(x)

def bayesian_rating(avg, n, C=3.8, m=50):
    if avg is None:
        avg = C
    if n is None:
        n = 0
    return (n / (n + m)) * avg + (m / (n + m)) * C


def pop_score(n, cap=1000):
    if not n or n <= 0:
        return 0.0
    return min(1.0, math.log10(n + 1) / math.log10(cap + 1))


def dist_score(km):
    if km <= 0:
        return 1.0
    if km >= MAX_KM:
        return 0.0
    return 1.0 - (km / MAX_KM)

def price_score(p_item, p_pref):
    if p_pref is None or p_item is None:
        return 0.75

    if p_item == p_pref:
        return 1.0

    if abs(p_item - p_pref) == 1:
        return 0.5

    return 0.0


def pagerank_score(pr):
    if pr is None or pr <= 0:
        return 0.0
    return min(1.0, math.log10(pr + 1))


def score_item(item: dict, base_prefs: Set[str], expanded_prefs: Set[str], raw_prefs: Set[str], price_pref: int | None, collab_score: float, pagerank: float, sentiment_score: float):
    tokens = {t.strip().lower() for t in item.get("categories", "").split(",") if t}
    
    # Separate matches for base and expanded preferences
    matched_base = tokens & base_prefs
    
    # Expanded prefs that are not in the base set
    additional_prefs = expanded_prefs - base_prefs
    matched_additional = tokens & additional_prefs

    # For the response, show all matched preferences
    matched = sorted(list(tokens & expanded_prefs))

    # --- Name Matching ---
    name_score = 0.0
    item_name_lower = item.get("name", "").lower()
    
    # Check both MAPPED prefs (e.g. "turkish") and RAW prefs (e.g. "kebap")
    # This fixes the issue where "kebap" maps to "turkish" and we lose the specificity
    matched_raw_name = {p for p in raw_prefs if p in item_name_lower}
    matched_mapped_name = {p for p in base_prefs if p in item_name_lower}
    
    if matched_raw_name or matched_mapped_name:
        name_score = 1.0
        # Add name-matched prefs to the response if they weren't found in categories
        for p in (matched_raw_name | matched_mapped_name):
            if p not in matched:
                matched.append(p)
    # --- End Name Matching ---

    # --- Review Text Matching ---
    review_score = 0.0
    reviews_text_lower = item.get("reviews_text", "").lower() if item.get("reviews_text") else ""
    
    matched_reviews = {p for p in raw_prefs if p in reviews_text_lower}
    
    if matched_reviews:
        review_score = 1.0
        # Add review-matched prefs to matched list for UI feedback
        for p in matched_reviews:
            if p not in matched:
                matched.append(p)
    
    matched.sort()
    # --- End Review Text Matching ---

    # Calculate scores
    base_score = (len(matched_base) / len(base_prefs)) if base_prefs else 0.0
    expanded_score = (len(matched_additional) / len(additional_prefs)) if additional_prefs else 0.0

    r = safe_float(bayesian_rating(item.get("rating_avg"), item.get("rating_count")) / 5.0)
    p = safe_float(pop_score(item.get("rating_count")))
    c = safe_float(0.8 * base_score + 0.2 * expanded_score) # Weighted category score
    d = safe_float(dist_score(item.get("km", MAX_KM)))
    pr_score = safe_float(price_score(item.get("price_range"), price_pref)) # Renamed to avoid conflict
    n_score = safe_float(name_score)
    rv_score = safe_float(review_score)
    cs = safe_float(collab_score)
    prs = safe_float(pagerank_score(pagerank))
    
    # Normalize sentiment_score from -1 to 1 range to 0 to 1 range for weighting
    normalized_sentiment = (sentiment_score + 1.0) / 2.0 
    
    score = safe_float(
        W_RATING * r +
        W_POP    * p +
        W_CAT    * c +
        W_DIST   * d +
        W_PRICE  * pr_score + 
        W_NAME   * n_score +
        W_REVIEW * rv_score +
        W_COLLAB * cs +
        W_PAGERANK * prs +
        W_SENTIMENT * normalized_sentiment
    )

    score_breakdown = {
        "weighted_rating": round(W_RATING * r, 6),
        "weighted_popularity": round(W_POP * p, 6),
        "weighted_category": round(W_CAT * c, 6),
        "weighted_distance": round(W_DIST * d, 6),
        "weighted_price": round(W_PRICE * pr_score, 6),
        "weighted_name_match": round(W_NAME * n_score, 6),
        "weighted_review_match": round(W_REVIEW * rv_score, 6),
        "weighted_collaborative": round(W_COLLAB * cs, 6),
        "weighted_pagerank": round(W_PAGERANK * prs, 6),
        "weighted_sentiment": round(W_SENTIMENT * normalized_sentiment, 6),
    }

    return round(score, 6), matched, sentiment_score, score_breakdown


# -------------------------------------------------
# Graph-based enhancements
# -------------------------------------------------

@lru_cache(maxsize=1)
def get_category_cooccurrence(min_freq: int = 5) -> Dict[str, List[str]]:
    query = """
    MATCH (c1:Category)<-[:HAS_CATEGORY]-(r:Restaurant)-[:HAS_CATEGORY]->(c2:Category)
    WHERE c1 <> c2
    RETURN c1.name AS base, c2.name AS related, count(*) AS freq
    ORDER BY freq DESC
    """
    co_map: Dict[str, List[str]] = {}

    with neo_driver.session() as session:
        for r in session.run(query):
            if r["freq"] >= min_freq:
                co_map.setdefault(r["base"], []).append(r["related"])

    return co_map


def expand_categories(prefs: Set[str], co_map: Dict[str, List[str]]) -> Set[str]:
    expanded = set(prefs)
    for p in prefs:
        for rel in co_map.get(p, []):
            expanded.add(rel)
    return expanded


def get_similar_restaurant_ids(seed_id: int, limit: int = 20) -> Set[int]:
    query = """
    MATCH (r:Restaurant {id: $rid})-[:HAS_CATEGORY]->(c)<-[:HAS_CATEGORY]-(o:Restaurant)
    WHERE r <> o
    WITH o, count(c) AS shared
    MATCH (r)-[:LOCATED_AT]->(l)<-[:LOCATED_AT]-(o)
    RETURN o.id AS id
    ORDER BY shared DESC
    LIMIT $limit
    """
    with neo_driver.session() as session:
        return {r["id"] for r in session.run(query, rid=seed_id, limit=limit)}


def get_similar_users(user_id: int, min_common: int = 2) -> List[int]:
    query = """
    MATCH (u1:User {id: $uid})-[:VISITED]->(r)<-[:VISITED]-(u2:User)
    WHERE u1 <> u2
    WITH u2, count(r) AS shared_restaurants
    WHERE shared_restaurants >= $min_common
    RETURN u2.id AS id
    ORDER BY shared_restaurants DESC
    LIMIT 100
    """
    with neo_driver.session() as session:
        return [r["id"] for r in session.run(query, uid=user_id, min_common=min_common)]


def get_user_recommendations(user_ids: List[int], min_rating: int = 4) -> Dict[int, float]:
    query = """
    MATCH (u:User)-[v:VISITED]->(r:Restaurant)
    WHERE u.id IN $uids AND v.rating >= $min_rating
    RETURN r.id AS id, count(*) AS score
    ORDER BY score DESC
    LIMIT 100
    """
    with neo_driver.session() as session:
        return {r["id"]: r["score"] for r in session.run(query, uids=user_ids, min_rating=min_rating)}


def get_user_preferences(user_id: int) -> Set[str]:
    query = """
    MATCH (u:User {id: $uid})-[:VISITED]->(r:Restaurant)-[:HAS_CATEGORY]->(c:Category)
    RETURN c.name AS category, count(c) AS count
    ORDER BY count DESC
    LIMIT 5
    """
    with neo_driver.session() as session:
        return {r["category"] for r in session.run(query, uid=user_id)}

# -------------------------------------------------
# API Endpoints
# -------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "algo": "graph_hybrid_v3"}


@app.get("/users")
def get_users():
    """
    Reads users from the data/users.xlsx file and returns them as a list.
    This is used to populate the user selection dropdown in the UI.
    """
    try:
        df_users = pd.read_excel(USERS_FILE_PATH)
        users = df_users.to_dict(orient="records")
        return users
    except FileNotFoundError:
        return {"error": "users.xlsx not found"}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}


@app.get("/rank", deprecated=True)
def rank(
    lat: float,
    lng: float,
    prefs: str,
    radius_km: float = 5.0,
    topK: int = 10,
    seed_restaurant_id: int | None = None,
    price_range: int | None = None
):
    # -----------------------------
    # 1. Parse and map preferences
    # -----------------------------
    raw_prefs = {p.strip().lower() for p in prefs.split(",") if p.strip()}
    base_prefs = {FOOD_TO_CATEGORY.get(p, p) for p in raw_prefs}

    # -----------------------------
    # 2. Category co-occurrence expansion
    # -----------------------------
    co_map = get_category_cooccurrence()
    expanded_prefs = expand_categories(base_prefs, co_map)

    # -----------------------------
    # 3. Get location-based candidates and pre-filter
    # -----------------------------
    all_candidates = get_candidates(
        lat=lat,
        lng=lng,
        radius_km=radius_km,
        raw_prefs=list(raw_prefs)
    )

    candidates = []
    if base_prefs:
        for c in all_candidates:
            tokens = {t.strip().lower() for t in c.get("categories", "").split(",") if t}
            if tokens & base_prefs:
                candidates.append(c)
    else:
        # If no prefs, use all candidates
        candidates = all_candidates


    # -----------------------------
    # 4. Graph similarity filtering (optional)
    # -----------------------------
    similar_ids = set()
    if seed_restaurant_id is not None:
        similar_ids = get_similar_restaurant_ids(seed_restaurant_id)

    # -----------------------------
    # 5. Scoring
    # -----------------------------
    items = []
    for c in candidates:
        if similar_ids and c["id"] not in similar_ids:
            continue

        # --- Analyze sentiment from reviews ---
        sentiment_score = analyze_sentiment(c.get("reviews_text", ""))
        sentiment_label = get_sentiment_label(sentiment_score)
        # --- End sentiment analysis ---

        score, matched, _, score_breakdown = score_item(
            c,
            base_prefs,
            expanded_prefs,
            raw_prefs,
            None, # price_pref
            0.0, # collab
            c.get("pagerank", 0.0),
            sentiment_score
        )

        items.append({
            "id": c["id"],
            "name": c["name"],
            "lat": c["lat"],
            "lng": c["lng"],
            "km": safe_float(round(c.get("km", 0.0), 3)),
            "categories": c["categories"],
            "rating_avg": c["rating_avg"],
            "rating_count": c["rating_count"],
            "price_range": c["price_range"],
            "pagerank": c["pagerank"],
            "matched_categories": matched,
            "sentiment_score": sentiment_score, # Add sentiment score to response
            "sentiment_label": sentiment_label, # Add sentiment label to response
            "score": score,
            "score_breakdown": score_breakdown
        })

    items.sort(key=lambda x: x["score"], reverse=True)
    items = items[:topK]

    # -----------------------------
    # 6. Response
    # -----------------------------
    return {
        "algo": "graph_hybrid_v3_similarity_enriched",
        "input_prefs": list(base_prefs),
        "expanded_prefs": list(expanded_prefs),
        "seed_restaurant_id": seed_restaurant_id,
        "count": len(items),
        "items": items
    }


@app.get("/users/{user_id}/recommendations")
def user_recommendations(
    user_id: int,
    lat: float,
    lng: float,
    radius_km: float = 5.0,
    topK: int = 10,
    price_range: int | None = None,
    prefs: str | None = None
):
    # -----------------------------
    # 1. Get user's historical preferences for SCORING
    # -----------------------------
    base_prefs = get_user_preferences(user_id)

    # -----------------------------
    # 2. Parse query preferences for FILTERING
    # -----------------------------
    query_prefs = set()
    raw_prefs_set = set()
    if prefs:
        raw_prefs_set = {p.strip().lower() for p in prefs.split(",") if p.strip()}
        query_prefs = {FOOD_TO_CATEGORY.get(p, p) for p in raw_prefs_set}

    # -----------------------------
    # 3. Category co-occurrence expansion (for scoring)
    # -----------------------------
    co_map = get_category_cooccurrence()
    expanded_prefs = expand_categories(base_prefs, co_map)

    # -----------------------------
    # 4. Collaborative filtering
    # -----------------------------
    similar_users = get_similar_users(user_id)
    collab_recs = get_user_recommendations(similar_users)

    # -----------------------------
    # 5. Get location-based candidates
    # -----------------------------
    all_candidates = get_candidates(
        lat=lat,
        lng=lng,
        radius_km=radius_km,
        raw_prefs=list(raw_prefs_set)
    )

    # -----------------------------
    # 6. Filter candidates by query_prefs if provided
    # -----------------------------
    candidates = []
    if query_prefs:
        for c in all_candidates:
            tokens = {t.strip().lower() for t in c.get("categories", "").split(",") if t}
            if tokens & query_prefs:
                candidates.append(c)
    else:
        candidates = all_candidates

    # -----------------------------
    # 7. Scoring
    # -----------------------------
    items = []
    for c in candidates:
        collab_score = collab_recs.get(c["id"], 0.0)
        
        # --- Analyze sentiment from reviews ---
        sentiment_score = analyze_sentiment(c.get("reviews_text", ""))
        sentiment_label = get_sentiment_label(sentiment_score)
        # --- End sentiment analysis ---

        score, matched, _, score_breakdown = score_item(
            c,
            base_prefs,
            expanded_prefs,
            raw_prefs_set,
            price_range,
            collab_score,
            c.get("pagerank", 0.0),
            sentiment_score # Pass sentiment score to score_item
        )

        items.append({
            "id": c["id"],
            "name": c["name"],
            "lat": c["lat"],
            "lng": c["lng"],
            "km": safe_float(round(c.get("km", 0.0), 3)),
            "categories": c["categories"],
            "rating_avg": c["rating_avg"],
            "rating_count": c["rating_count"],
            "price_range": c["price_range"],
            "pagerank": c["pagerank"],
            "matched_categories": matched,
            "sentiment_score": sentiment_score, # Add sentiment score to response
            "sentiment_label": sentiment_label, # Add sentiment label to response
            "score": score,
            "score_breakdown": score_breakdown
        })

    items.sort(key=lambda x: x["score"], reverse=True)
    items = items[:topK]

    # -----------------------------
    # 8. Response
    # -----------------------------
    return {
        "algo": "graph_hybrid_v6_personalized_filtered",
        "user_historical_prefs": list(base_prefs),
        "query_prefs": list(query_prefs),
        "expanded_prefs": list(expanded_prefs),
        "user_id": user_id,
        "count": len(items),
        "items": items
    }
