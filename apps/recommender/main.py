from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from typing import Set, List, Dict
import math

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

NEO4J_URI = "bolt://neo4j:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "oburus_pass"

neo_driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS)
)

# -------------------------------------------------
# Hybrid scoring weights
# -------------------------------------------------

W_RATING = 0.45
W_POP    = 0.15
W_CAT    = 0.25
W_DIST   = 0.15

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


def score_item(item: dict, prefs: Set[str]):
    tokens = {t.strip().lower() for t in item.get("categories", "").split(",") if t}
    matched = sorted(tokens & prefs)

    r = safe_float(bayesian_rating(item.get("rating_avg"), item.get("rating_count")) / 5.0)
    p = safe_float(pop_score(item.get("rating_count")))
    c = safe_float((len(matched) / len(prefs)) if prefs else 0.0)
    d = safe_float(dist_score(item.get("km", MAX_KM)))

    score = safe_float(
        W_RATING * r +
        W_POP    * p +
        W_CAT    * c +
        W_DIST   * d
    )

    return round(score, 6), matched


# -------------------------------------------------
# Graph-based enhancements
# -------------------------------------------------

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

# -------------------------------------------------
# API Endpoints
# -------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "algo": "graph_hybrid_v3"}


@app.get("/rank")
def rank(
    lat: float,
    lng: float,
    prefs: str,
    radius_km: float = 5.0,
    topK: int = 10,
    seed_restaurant_id: int | None = None
):
    # -----------------------------
    # 1. Parse preferences
    # -----------------------------
    base_prefs = {p.strip().lower() for p in prefs.split(",") if p.strip()}

    # -----------------------------
    # 2. Category co-occurrence expansion
    # -----------------------------
    co_map = get_category_cooccurrence()
    expanded_prefs = expand_categories(base_prefs, co_map)

    # -----------------------------
    # 3. Get location-based candidates
    # -----------------------------
    candidates = get_candidates(
        lat=lat,
        lng=lng,
        radius_km=radius_km
    )

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

        score, matched = score_item(c, expanded_prefs)

        items.append({
            "id": c["id"],
            "name": c["name"],
            "lat": c["lat"],
            "lng": c["lng"],
            "km": safe_float(round(c.get("km", 0.0), 3)),
            "categories": c["categories"],
            "rating_avg": c["rating_avg"],
            "rating_count": c["rating_count"],
            "matched_prefs": matched,
            "score": score
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
