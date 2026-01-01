import pandas as pd
import random
import os

# Configuration
NUM_USERS = 200
VISITS_PER_USER_MIN = 15
VISITS_PER_USER_MAX = 25
OUTPUT_DIR = "data"
RESTAURANTS_FILE = os.path.join(OUTPUT_DIR, "restaurants.xlsx")
USERS_OUTPUT = os.path.join(OUTPUT_DIR, "synthetic_users.xlsx")
VISITS_OUTPUT = os.path.join(OUTPUT_DIR, "synthetic_visits.xlsx")

# Turkish Names Data
FIRST_NAMES = [
    "Ahmet", "Mehmet", "Mustafa", "Ali", "Hüseyin", "Hasan", "İbrahim", "İsmail", "Osman", "Yusuf",
    "Ayşe", "Fatma", "Emine", "Hatice", "Zeynep", "Elif", "Meryem", "Şerife", "Zehra", "Sultan",
    "Can", "Cem", "Deniz", "Ege", "Umut", "Barış", "Burak", "Berk", "Kerem", "Mert",
    "Selin", "Sena", "Buse", "Gizem", "Gamze", "Kübra", "Büşra", "Esra", "Seda", "Sinem"
]

LAST_NAMES = [
    "Yılmaz", "Kaya", "Demir", "Çelik", "Şahin", "Yıldız", "Yıldırım", "Öztürk", "Aydın", "Özdemir",
    "Arslan", "Doğan", "Kılıç", "Aslan", "Çetin", "Kara", "Koç", "Kurt", "Özkan", "Şimşek"
]

# Personas and their preferred categories
PERSONAS = {
    "Kebap Lover": {
        "prefs": ["turkish", "kebap", "steakhouse"],
        "weight": 0.3
    },
    "Cafe Addict": {
        "prefs": ["coffee", "cafe", "dessert", "bakery"],
        "weight": 0.2
    },
    "Fast Food Fan": {
        "prefs": ["fast_food", "burger", "pizza"],
        "weight": 0.2
    },
    "Seafood/Med": {
        "prefs": ["seafood", "mediterranean", "italian"],
        "weight": 0.1
    },
    "General": {
        "prefs": [], # No specific preference
        "weight": 0.2
    }
}

# TripAdvisor cuisine mapping (simplified for matching)
CUISINE_MAP = {
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
    "steakhouse": "steakhouse",
    "international": "restaurant"
}

def generate_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def get_categories(cuisine_str):
    if pd.isna(cuisine_str):
        return []
    raw = [c.strip().lower() for c in str(cuisine_str).split(",")]
    mapped = []
    for r in raw:
        if r in CUISINE_MAP:
            mapped.append(CUISINE_MAP[r])
        else:
            mapped.append("other")
    return mapped

def main():
    print(f"Loading restaurants from {RESTAURANTS_FILE}...")
    try:
        df_rest = pd.read_excel(RESTAURANTS_FILE)
    except FileNotFoundError:
        print("❌ Error: restaurants.xlsx not found. Please ensure it exists in 'data/' folder.")
        return

    # Prepare restaurant pool with categories
    restaurant_pool = []
    for _, row in df_rest.iterrows():
        cats = get_categories(row.get("tripadvisor_cuisine"))
        restaurant_pool.append({
            "name": row["name"],
            "categories": set(cats)
        })
    
    print(f"Loaded {len(restaurant_pool)} restaurants.")

    # Generate Users
    users = []
    visits = []
    
    # Start IDs from 1000 to avoid conflict with existing small data
    start_user_id = 1000
    
    print(f"Generating {NUM_USERS} users with personas...")

    for i in range(NUM_USERS):
        uid = start_user_id + i
        name = generate_name()
        
        # Assign Persona
        persona_name = random.choices(
            list(PERSONAS.keys()), 
            weights=[p["weight"] for p in PERSONAS.values()]
        )[0]
        persona_prefs = set(PERSONAS[persona_name]["prefs"])
        
        users.append({"user_id": uid, "name": name, "persona": persona_name}) # Persona is for debug/analysis, not schema

        # Generate Visits
        num_visits = random.randint(VISITS_PER_USER_MIN, VISITS_PER_USER_MAX)
        
        # Filter restaurants matching persona
        preferred_pool = []
        other_pool = []
        
        if persona_name == "General":
            other_pool = restaurant_pool
        else:
            for r in restaurant_pool:
                if r["categories"] & persona_prefs:
                    preferred_pool.append(r)
                else:
                    other_pool.append(r)
        
        # Select restaurants
        # Mix: 70% preferred, 30% random (if applicable)
        visited_restaurants = set()
        
        for _ in range(num_visits):
            # Decide pool
            use_preferred = False
            if preferred_pool and random.random() < 0.7:
                use_preferred = True
            elif not other_pool: # Fallback if no others
                use_preferred = True
            
            # Pick
            pool = preferred_pool if use_preferred else other_pool
            if not pool:
                pool = restaurant_pool # Fallback to all
            
            r = random.choice(pool)
            
            # Avoid duplicates for same user
            if r["name"] in visited_restaurants:
                continue
            visited_restaurants.add(r["name"])
            
            # Rating Logic
            # If matches persona -> High rating (4-5)
            # If random -> Gaussian around 3.5
            rating = 3
            if use_preferred:
                rating = random.choices([4, 5], weights=[0.4, 0.6])[0]
            else:
                rating = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.1, 0.2, 0.35, 0.3])[0]
            
            visits.append({
                "user_id": uid, 
                "restaurant_name": r["name"], 
                "rating": rating
            })

    # Convert to DataFrame
    df_users = pd.DataFrame(users)
    df_visits = pd.DataFrame(visits)
    
    # Save Users (only user_id, name columns to match schema)
    df_users[["user_id", "name"]].to_excel(USERS_OUTPUT, index=False)
    print(f"✅ Saved {len(df_users)} users to {USERS_OUTPUT}")
    
    # Save Visits
    df_visits.to_excel(VISITS_OUTPUT, index=False)
    print(f"✅ Saved {len(df_visits)} visits to {VISITS_OUTPUT}")
    
    print("\nNext Steps:")
    print("1. Run this script to generate the files.")
    print("2. Run 'ingest_to_graph.py' to load this new data into Neo4j.")
    print("   (Note: You might want to update ingest_to_graph.py to point to these new files or rename them)")

if __name__ == "__main__":
    main()
