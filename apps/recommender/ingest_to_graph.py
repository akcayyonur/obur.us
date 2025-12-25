import os
import pandas as pd
from neo4j import GraphDatabase

# -------------------------------------------------
# Neo4j connection
# -------------------------------------------------
NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "oburus_pass")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

XLSX_PATH = "/app/data/restaurants.xlsx"

# -------------------------------------------------
# TripAdvisor cuisine → OBURUS category ontology
# -------------------------------------------------
CUISINE_TO_CATEGORY = {
    "cafe": "coffee",
    "coffee": "coffee",
    "desserts": "dessert",
    "bakery": "dessert",
    "turkish": "turkish",
    "mediterranean": "mediterranean",
    "italian": "italian",
    "asian": "asian",
    "fast food": "fast_food",
    "pizza": "pizza",
    "seafood": "seafood",
    "bar": "bar",
    "international": "restaurant",
}

def parse_cuisines(raw):
    if raw is None or pd.isna(raw):
        return []
    return [c.strip().lower() for c in str(raw).split(",") if c.strip()]

def map_cuisines_to_categories(cuisines):
    cats = {CUISINE_TO_CATEGORY[c] for c in cuisines if c in CUISINE_TO_CATEGORY}
    if not cats:
        cats = {"other"}
    return sorted(cats)

# -------------------------------------------------
# Neo4j ingestion (TEK QUERY, TEK MERGE)
# -------------------------------------------------
def ingest(tx, payload):
    tx.run(
        """
        MERGE (r:Restaurant {id: $id})
        SET r.name = $name,
            r.rating_avg = $rating_avg,
            r.rating_count = $rating_count,
            r.source = 'tripadvisor'
        WITH r
        MERGE (l:Location {lat: $lat, lng: $lng})
        MERGE (r)-[:LOCATED_AT]->(l)
        WITH r
        UNWIND $categories AS cat
        MERGE (c:Category {name: cat})
        MERGE (r)-[:HAS_CATEGORY]->(c)
        """,
        **payload
    )

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    df = pd.read_excel(XLSX_PATH)

    # --- Robust cleaning ---
    df["name"] = df["name"].astype("string").str.strip()
    df.loc[df["name"] == "", "name"] = pd.NA

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Zorunlu alanlar
    df = df.dropna(subset=["restaurant_id", "name", "latitude", "longitude"])

    # restaurant_id doğrulama
    if df["restaurant_id"].duplicated().any():
        raise ValueError("Duplicate restaurant_id bulundu!")

    print(f"✅ Geçerli satır sayısı: {len(df)}")
    print("✅ İlk 10 name örneği:", df["name"].head(10).tolist())

    with driver.session() as session:
        for _, row in df.iterrows():
            payload = {
                "id": int(row["restaurant_id"]),
                "name": str(row["name"]),
                "lat": float(row["latitude"]),
                "lng": float(row["longitude"]),
                "rating_avg": row.get("tripadvisor_rating"),
                "rating_count": int(row.get("tripadvisor_review_count", 0) or 0),
                "categories": map_cuisines_to_categories(
                    parse_cuisines(row.get("tripadvisor_cuisine"))
                ),
            }

            session.execute_write(ingest, payload)

    print("✅ Ingestion başarıyla tamamlandı.")

if __name__ == "__main__":
    main()
