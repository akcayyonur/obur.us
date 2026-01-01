from neo4j import GraphDatabase
import os

# -------------------------------------------------
# Neo4j connection
# -------------------------------------------------

NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "oburus_pass")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASS)
)

# -------------------------------------------------
# Category fallback (isimden çıkarım)
# -------------------------------------------------

def infer_categories_from_name(name: str) -> list[str]:
    n = name.lower()
    cats = []

    if any(k in n for k in ["kahve", "coffee", "cafe"]):
        cats.append("coffee")

    if any(k in n for k in ["tatl", "dessert", "baklava", "kunefe", "künefe"]):
        cats.append("dessert")

    if any(k in n for k in ["meyhane", "rakı", "meze", "taverna"]):
        cats.append("meyhane")

    if any(k in n for k in ["kebap", "ocakbaşı", "adana", "urfa"]):
        cats.append("kebap")

    if any(k in n for k in ["burger"]):
        cats.append("burger")

    if any(k in n for k in ["balık", "fish", "seafood"]):
        cats.append("seafood")

    if not cats:
        cats.append("other")

    return cats

# -------------------------------------------------
# Graph candidate generation
# -------------------------------------------------

def get_candidates(
    lat: float,
    lng: float,
    radius_km: float = 5.0,
    limit: int = 200,
    raw_prefs: list[str] = None
):
    """
    Generates restaurant candidates via Neo4j graph.
    Prioritizes restaurants matching 'raw_prefs' (intent) in name or reviews,
    then sorts by distance.
    """

    # Calculate distance in meters for the query
    radius_meters = radius_km * 1000.0
    
    # Prepare keywords for Cypher (lowercase for case-insensitive search if needed, 
    # though we'll handle lowercase in Cypher or Python logic. 
    # Here we assume reviews/names in DB are mixed case, so we use toLower() in Cypher).
    keywords = [p.lower() for p in (raw_prefs or []) if p.strip()]

    query = """
    MATCH (r:Restaurant)-[:LOCATED_AT]->(l:Location)
    WITH r, l, point.distance(point({latitude: l.lat, longitude: l.lng}), point({latitude: $lat, longitude: $lng})) AS dist_meters
    WHERE dist_meters < $radius_meters
    
    // Determine if this restaurant matches the user's specific intent
    WITH r, l, dist_meters,
         reduce(match_found = false, k IN $keywords | 
            match_found OR toLower(r.name) CONTAINS k OR toLower(r.reviews_text) CONTAINS k
         ) AS is_intent_match

    OPTIONAL MATCH (r)-[:HAS_CATEGORY]->(c:Category)
    
    RETURN
        r.id            AS id,
        r.name          AS name,
        r.rating_avg    AS rating_avg,
        r.rating_count  AS rating_count,
        r.price_range   AS price_range,
        r.pagerank      AS pagerank,
        r.reviews_text  AS reviews_text,
        l.lat           AS lat,
        l.lng           AS lng,
        dist_meters / 1000.0 AS km,
        collect(c.name) AS categories,
        is_intent_match
    
    // Sort primarily by Intent Match (true > false), then by Distance
    ORDER BY is_intent_match DESC, km ASC
    LIMIT $limit
    """

    results = []

    with driver.session() as session:
        # Pass parameters safely
        rows = session.run(query, lat=lat, lng=lng, radius_meters=radius_meters, limit=limit, keywords=keywords)

        for row in rows:
            # 1️⃣ Graph categories
            cats = [c for c in row["categories"] if c]

            # 2️⃣ Fallback: infer category from name
            if not cats:
                cats = infer_categories_from_name(row["name"])

            results.append({
                "id": row["id"],
                "name": row["name"],
                "lat": row["lat"],
                "lng": row["lng"],
                "categories": ",".join(cats),
                "rating_avg": row["rating_avg"],
                "rating_count": row["rating_count"],
                "price_range": row["price_range"],
                "pagerank": row["pagerank"],
                "reviews_text": row["reviews_text"],
                "km": row["km"],
                # We can optionally pass 'is_intent_match' if we want to debug, 
                # but the Python scoring will re-evaluate strict matches anyway.
            })

    return results
