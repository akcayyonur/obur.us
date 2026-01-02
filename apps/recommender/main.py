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
import json
from nlp_processor import analyze_sentiment, get_sentiment_label
from graph_candidates import get_candidates
from neo4j import GraphDatabase

# -------------------------------------------------
# Custom JSON encoder to handle NaN/Inf
# -------------------------------------------------
class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,  # This will raise an error if NaN/Inf found
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

# -------------------------------------------------
# App
# -------------------------------------------------

app = FastAPI(
    title="OBURUS Graph-Based Personalized Recommender",
    version="3.0",
    default_response_class=SafeJSONResponse
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
# Exception Handlers
# -------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return SafeJSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return SafeJSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    import traceback
    return SafeJSONResponse(
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
W_NAME   = 0.20
W_REVIEW = 0.30
W_COLLAB = 0.10
W_PAGERANK = 0.18
W_SENTIMENT = 0.10

MAX_KM = 20.0

# -------------------------------------------------
# Helper functions (Scoring) - WITH NaN/Inf PROTECTION
# -------------------------------------------------

def safe_float(x, default=0.0):
    """Convert to float and ensure it's JSON-serializable (no NaN/Inf)"""
    if x is None:
        return default
    try:
        result = float(x)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default

def bayesian_rating(avg, n, C=3.8, m=50):
    avg = safe_float(avg, C)
    n = safe_float(n, 0)
    if n <= 0:
        return C
    return (n / (n + m)) * avg + (m / (n + m)) * C


def pop_score(n, cap=1000):
    n = safe_float(n, 0)
    if n <= 0:
        return 0.0
    return min(1.0, math.log10(n + 1) / math.log10(cap + 1))


def dist_score(km):
    km = safe_float(km, MAX_KM)
    if km <= 0:
        return 1.0
    if km >= MAX_KM:
        return 0.0
    return 1.0 - (km / MAX_KM)

def price_score(p_item, p_pref):
    if p_pref is None or p_item is None:
        return 0.75

    p_item = safe_float(p_item, 0)
    p_pref = safe_float(p_pref, 0)

    if p_item == p_pref:
        return 1.0

    if abs(p_item - p_pref) == 1:
        return 0.5

    return 0.0


def pagerank_score(pr):
    pr = safe_float(pr, 0)
    if pr <= 0:
        return 0.0
    return min(1.0, math.log10(pr + 1))


def sanitize_value(value):
    """Ensure a value is JSON-serializable"""
    if value is None:
        return None
    if isinstance(value, (int, str, bool)):
        return value
    if isinstance(value, float):
        return safe_float(value, 0.0)
    if isinstance(value, list):
        return [sanitize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: sanitize_value(v) for k, v in value.items()}
    return str(value)


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
    
    matched_raw_name = {p for p in raw_prefs if p in item_name_lower}
    matched_mapped_name = {p for p in base_prefs if p in item_name_lower}
    
    if matched_raw_name or matched_mapped_name:
        name_score = 1.0
        for p in (matched_raw_name | matched_mapped_name):
            if p not in matched:
                matched.append(p)

    # --- Review Text Matching ---
    review_score = 0.0
    reviews_text_lower = item.get("reviews_text", "").lower() if item.get("reviews_text") else ""
    
    matched_reviews = {p for p in raw_prefs if p in reviews_text_lower}
    
    if matched_reviews:
        review_score = 1.0
        for p in matched_reviews:
            if p not in matched:
                matched.append(p)
    
    matched.sort()

    # Calculate scores with safe_float
    base_score = (len(matched_base) / len(base_prefs)) if base_prefs else 0.0
    expanded_score = (len(matched_additional) / len(additional_prefs)) if additional_prefs else 0.0

    r = safe_float(bayesian_rating(item.get("rating_avg"), item.get("rating_count")) / 5.0)
    p = safe_float(pop_score(item.get("rating_count")))
    c = safe_float(0.8 * base_score + 0.2 * expanded_score)
    d = safe_float(dist_score(item.get("km", MAX_KM)))
    pr_score = safe_float(price_score(item.get("price_range"), price_pref))
    n_score = safe_float(name_score)
    rv_score = safe_float(review_score)
    cs = safe_float(collab_score)
    prs = safe_float(pagerank_score(pagerank))
    
    # Normalize sentiment_score from -1 to 1 range to 0 to 1 range for weighting
    sentiment_score = safe_float(sentiment_score, 0.0)
    normalized_sentiment = safe_float((sentiment_score + 1.0) / 2.0)
    
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
        "weighted_rating": safe_float(W_RATING * r, 0.0),
        "weighted_popularity": safe_float(W_POP * p, 0.0),
        "weighted_category": safe_float(W_CAT * c, 0.0),
        "weighted_distance": safe_float(W_DIST * d, 0.0),
        "weighted_price": safe_float(W_PRICE * pr_score, 0.0),
        "weighted_name_match": safe_float(W_NAME * n_score, 0.0),
        "weighted_review_match": safe_float(W_REVIEW * rv_score, 0.0),
        "weighted_collaborative": safe_float(W_COLLAB * cs, 0.0),
        "weighted_pagerank": safe_float(W_PAGERANK * prs, 0.0),
        "weighted_sentiment": safe_float(W_SENTIMENT * normalized_sentiment, 0.0),
    }

    return score, matched, sentiment_score, score_breakdown


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
    Reads users from the data/synthetic_users.xlsx file and returns them as a list.
    This is used to populate the user selection dropdown in the UI.
    """
    try:
        df_users = pd.read_excel(USERS_FILE_PATH)
        # Remove duplicates based on user_id
        df_users = df_users.drop_duplicates(subset=['user_id'], keep='first')
        users = df_users.to_dict(orient="records")
        return users
    except FileNotFoundError:
        return {"error": "synthetic_users.xlsx not found"}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}


@app.get("/rank", deprecated=True)
def rank(
    lat: float,
    lng: float,
    prefs: str = "",
    radius_km: float = 5.0,
    topK: int = 10,
    seed_restaurant_id: int | None = None,
    price_range: int | None = None
):
    # Parse and map preferences
    raw_prefs = {p.strip().lower() for p in prefs.split(",") if p.strip()}
    base_prefs = {FOOD_TO_CATEGORY.get(p, p) for p in raw_prefs}

    # Category co-occurrence expansion
    co_map = get_category_cooccurrence()
    expanded_prefs = expand_categories(base_prefs, co_map)

    # Get location-based candidates
    all_candidates = get_candidates(
        lat=lat,
        lng=lng,
        radius_km=radius_km,
        raw_prefs=list(raw_prefs)
    )

    # Pre-filter candidates if prefs provided
    candidates = []
    if base_prefs:
        for c in all_candidates:
            tokens = {t.strip().lower() for t in c.get("categories", "").split(",") if t}
            if tokens & base_prefs:
                candidates.append(c)
    else:
        candidates = all_candidates

    # Graph similarity filtering (optional)
    similar_ids = set()
    if seed_restaurant_id is not None:
        similar_ids = get_similar_restaurant_ids(seed_restaurant_id)

    # Scoring
    items = []
    for c in candidates:
        if similar_ids and c["id"] not in similar_ids:
            continue

        sentiment_score = analyze_sentiment(c.get("reviews_text", ""))
        sentiment_label = get_sentiment_label(sentiment_score)

        score, matched, _, score_breakdown = score_item(
            c,
            base_prefs,
            expanded_prefs,
            raw_prefs,
            None,
            0.0,
            c.get("pagerank", 0.0),
            sentiment_score
        )

        items.append({
            "id": sanitize_value(c["id"]),
            "name": sanitize_value(c["name"]),
            "lat": safe_float(c.get("lat"), 0.0),
            "lng": safe_float(c.get("lng"), 0.0),
            "km": safe_float(c.get("km", 0.0)),
            "categories": sanitize_value(c.get("categories")),
            "rating_avg": safe_float(c.get("rating_avg")),
            "rating_count": safe_float(c.get("rating_count"), 0),
            "price_range": safe_float(c.get("price_range"), 0),
            "pagerank": safe_float(c.get("pagerank"), 0.0),
            "matched_categories": matched,
            "sentiment_score": safe_float(sentiment_score, 0.0),
            "sentiment_label": sentiment_label,
            "score": safe_float(score, 0.0),
            "score_breakdown": score_breakdown
        })

    items.sort(key=lambda x: x["score"], reverse=True)
    items = items[:topK]

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
    # Get user's historical preferences for SCORING
    base_prefs = get_user_preferences(user_id)

    # Parse query preferences for FILTERING
    query_prefs = set()
    raw_prefs_set = set()
    if prefs:
        raw_prefs_set = {p.strip().lower() for p in prefs.split(",") if p.strip()}
        query_prefs = {FOOD_TO_CATEGORY.get(p, p) for p in raw_prefs_set}

    # Category co-occurrence expansion (for scoring)
    co_map = get_category_cooccurrence()
    expanded_prefs = expand_categories(base_prefs, co_map)

    # Collaborative filtering
    similar_users = get_similar_users(user_id)
    collab_recs = get_user_recommendations(similar_users)

    # Get location-based candidates
    all_candidates = get_candidates(
        lat=lat,
        lng=lng,
        radius_km=radius_km,
        raw_prefs=list(raw_prefs_set)
    )

    # Filter candidates by query_prefs if provided, otherwise use all
    candidates = []
    if query_prefs:
        for c in all_candidates:
            tokens = {t.strip().lower() for t in c.get("categories", "").split(",") if t}
            if tokens & query_prefs:
                candidates.append(c)
    else:
        # If no query prefs, use all candidates and rank by user's historical preferences
        candidates = all_candidates

    # Scoring
    items = []
    for c in candidates:
        collab_score = collab_recs.get(c["id"], 0.0)
        
        sentiment_score = analyze_sentiment(c.get("reviews_text", ""))
        sentiment_label = get_sentiment_label(sentiment_score)

        score, matched, _, score_breakdown = score_item(
            c,
            base_prefs,
            expanded_prefs,
            raw_prefs_set,
            price_range,
            collab_score,
            c.get("pagerank", 0.0),
            sentiment_score
        )

        items.append({
            "id": sanitize_value(c["id"]),
            "name": sanitize_value(c["name"]),
            "lat": safe_float(c.get("lat"), 0.0),
            "lng": safe_float(c.get("lng"), 0.0),
            "km": safe_float(c.get("km", 0.0)),
            "categories": sanitize_value(c.get("categories")),
            "rating_avg": safe_float(c.get("rating_avg")),
            "rating_count": safe_float(c.get("rating_count"), 0),
            "price_range": safe_float(c.get("price_range"), 0),
            "pagerank": safe_float(c.get("pagerank"), 0.0),
            "matched_categories": matched,
            "sentiment_score": safe_float(sentiment_score, 0.0),
            "sentiment_label": sentiment_label,
            "score": safe_float(score, 0.0),
            "score_breakdown": score_breakdown
        })

    items.sort(key=lambda x: x["score"], reverse=True)
    items = items[:topK]

    return {
        "algo": "graph_hybrid_v6_personalized_filtered",
        "user_historical_prefs": list(base_prefs),
        "query_prefs": list(query_prefs),
        "expanded_prefs": list(expanded_prefs),
        "user_id": user_id,
        "count": len(items),
        "items": items
    }