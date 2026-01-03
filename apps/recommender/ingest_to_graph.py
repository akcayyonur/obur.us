import os
import pandas as pd
from neo4j import GraphDatabase

# -------------------------------------------------
# Neo4j connection
# -------------------------------------------------
NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "oburus_pass")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Define paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

XLSX_RESTAURANTS_PATH = os.getenv("XLSX_RESTAURANTS_PATH", os.path.join(DATA_DIR, "restaurants.xlsx"))
XLSX_USERS_PATH = os.getenv("XLSX_USERS_PATH", os.path.join(DATA_DIR, "synthetic_users.xlsx"))
XLSX_VISITS_PATH = os.getenv("XLSX_VISITS_PATH", os.path.join(DATA_DIR, "synthetic_visits.xlsx"))

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
def create_constraints(tx):
    # Unique constraint for Restaurant ID
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Restaurant) REQUIRE r.id IS UNIQUE")
    # Unique constraint for User ID
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
    # Index for Location coordinates (composite index)
    tx.run("CREATE INDEX IF NOT EXISTS FOR (l:Location) ON (l.lat, l.lng)")

def ingest_restaurants(tx, payload):
    tx.run(
        """
        MERGE (r:Restaurant {id: $id})
        SET r.name = $name,
            r.rating_avg = $rating_avg,
            r.rating_count = $rating_count,
            r.price_range = $price_range,
            r.source = 'tripadvisor',
            r.reviews_text = $reviews_text,
            r.website = $website
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

def ingest_users(tx, payload):
    tx.run(
        """
        MERGE (u:User {id: $id})
        SET u.name = $name
        """,
        **payload
    )

def ingest_visits(tx, payload):
    tx.run(
        """
        MATCH (u:User {id: $user_id})
        MATCH (r:Restaurant {id: $restaurant_id})
        MERGE (u)-[v:VISITED]->(r)
        SET v.rating = $rating
        """,
        **payload
    )

def calculate_pagerank(tx):
    # Drop the graph projection if it exists
    tx.run("""
    CALL gds.graph.drop('restaurants', false)
    """)
    
    # Create a graph projection for GDS
    tx.run("""
    CALL gds.graph.project(
        'restaurants',
        ['Restaurant', 'Category'],
        {
            HAS_CATEGORY: {
                orientation: 'UNDIRECTED'
            }
        }
    )
    """)

    # Run PageRank and write the results back to the nodes
    tx.run("""
    CALL gds.pageRank.write(
        'restaurants',
        {
            writeProperty: 'pagerank'
        }
    )
    """)

def calculate_fastrp_embeddings(tx):
    # Mutate the graph projection with FastRP embeddings
    tx.run("""
    CALL gds.fastRPEmbedding.mutate(
        'restaurants',
        {
            embeddingDimension: 256,
            mutateProperty: 'embedding'
        }
    )
    """)

    # Drop the graph projection
    tx.run("CALL gds.graph.drop('restaurants')")

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    # Create constraints and indexes
    with driver.session() as session:
        session.execute_write(create_constraints)
    print("✅ Constraints & Indexes oluşturuldu.")

    # Ingest Restaurants
    df_restaurants = pd.read_excel(XLSX_RESTAURANTS_PATH)

    # --- Robust cleaning ---
    df_restaurants["name"] = df_restaurants["name"].astype("string").str.strip()
    df_restaurants.loc[df_restaurants["name"] == "", "name"] = pd.NA

    df_restaurants["latitude"] = pd.to_numeric(df_restaurants["latitude"], errors="coerce")
    df_restaurants["longitude"] = pd.to_numeric(df_restaurants["longitude"], errors="coerce")
    df_restaurants["price_range"] = df_restaurants["tripadvisor_price_level"].astype(str).str.extract(r'(^\$+)')[0].str.len().fillna(0).astype(int)

    # Zorunlu alanlar
    df_restaurants = df_restaurants.dropna(subset=["restaurant_id", "name", "latitude", "longitude"])

    # restaurant_id doğrulama
    if df_restaurants["restaurant_id"].duplicated().any():
        raise ValueError("Duplicate restaurant_id bulundu!")

    print(f"✅ Geçerli satır sayısı (restaurants): {len(df_restaurants)}")
    print("✅ İlk 10 name örneği (restaurants):", df_restaurants["name"].head(10).tolist())

    restaurant_name_to_id = pd.Series(
        df_restaurants.restaurant_id.values, index=df_restaurants.name
    ).to_dict()

    social_domains = ["instagram.com", "facebook.com", "twitter.com", "tripadvisor", "yelp.com", "foursquare.com", "tripadvisor.com.tr", "tripadvisor.com"]

    with driver.session() as session:
        for _, row in df_restaurants.iterrows():
            raw_website = str(row.get("website", "")).strip()
            website = None
            if raw_website and raw_website.lower() != "nan" and raw_website.lower() != "none":
                is_social = any(domain in raw_website.lower() for domain in social_domains)
                if not is_social:
                    website = raw_website

            payload = {
                "id": int(row["restaurant_id"]),
                "name": str(row["name"]),
                "lat": float(row["latitude"]),
                "lng": float(row["longitude"]),
                "rating_avg": row.get("tripadvisor_rating"),
                "rating_count": int(row.get("tripadvisor_review_count", 0) or 0),
                "price_range": int(row.get("price_range", 0) or 0),
                "website": website,
                "categories": map_cuisines_to_categories(
                    parse_cuisines(row.get("tripadvisor_cuisine"))
                ),
                "reviews_text": " ||| ".join([
                    str(row[f"ta_review_{i}_text"]) for i in range(1, 6)
                    if pd.notna(row.get(f"ta_review_{i}_text")) and str(row.get(f"ta_review_{i}_text")).strip() != ""
                ])
            }

            session.execute_write(ingest_restaurants, payload)

    print("✅ Ingestion (restaurants) başarıyla tamamlandı.")

    # Ingest Users
    df_users = pd.read_excel(XLSX_USERS_PATH)
    print(f"✅ Geçerli satır sayısı (users): {len(df_users)}")

    with driver.session() as session:
        for _, row in df_users.iterrows():
            payload = {
                "id": int(row["user_id"]),
                "name": str(row["name"]),
            }
            session.execute_write(ingest_users, payload)

    print("✅ Ingestion (users) başarıyla tamamlandı.")

    # Ingest Visits
    df_visits = pd.read_excel(XLSX_VISITS_PATH)
    df_visits["restaurant_name"] = df_visits["restaurant_name"].str.strip()
    print(f"✅ Geçerli satır sayısı (visits): {len(df_visits)}")

    with driver.session() as session:
        for _, row in df_visits.iterrows():
            restaurant_name = row["restaurant_name"]
            restaurant_id = restaurant_name_to_id.get(restaurant_name)

            if restaurant_id is None:
                print(f"⚠️ Restaurant not found for name: '{restaurant_name}', skipping visit.")
                continue

            payload = {
                "user_id": int(row["user_id"]),
                "restaurant_id": int(restaurant_id),
                "rating": int(row["rating"]),
            }
            session.execute_write(ingest_visits, payload)

    print("✅ Ingestion (visits) başarıyla tamamlandı.")

    # Calculate PageRank
    with driver.session() as session:
        session.execute_write(calculate_pagerank)
    print("✅ PageRank calculation başarıyla tamamlandı.")
    
    # Calculate FastRP embeddings
    # with driver.session() as session:
    #     session.execute_write(calculate_fastrp_embeddings)
    # print("✅ FastRP embedding calculation başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
