from neo4j import GraphDatabase
import os
import math

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
# Geo helpers
# -------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

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
    limit: int = 200
):
    """
    Neo4j graph üzerinden restaurant adayları üretir.
    Category yoksa isimden fallback uygular.
    """

    query = """
    MATCH (r:Restaurant)-[:LOCATED_AT]->(l:Location)
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
        collect(c.name) AS categories
    """

    results = []

    with driver.session() as session:
        rows = session.run(query)

        for row in rows:
            km = haversine_km(
                lat, lng,
                row["lat"], row["lng"]
            )

            if km > radius_km:
                continue

            # 1️⃣ Graph kategorileri
            cats = [c for c in row["categories"] if c]

            # 2️⃣ Fallback: isimden kategori çıkar
            if not cats:
                cats = infer_categories_from_name(row["name"])
                print("DEBUG:", row["name"], "→ cats:", cats)


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
                "km": km
            })

    # Mesafeye göre sırala
    results.sort(key=lambda x: x["km"])

    return results[:limit]
