# apps/recommender/tests/test_recommender.py
import os
import pandas as pd
import time
import httpx

BASE = os.getenv("RECO_BASE", "http://localhost:8000")
DATA_DIR = "/app/data"
XLSX = f"{DATA_DIR}/restaurants.xlsx"

def _ensure_excel_with_rows():
    os.makedirs(DATA_DIR, exist_ok=True)
    # Minimal, temiz bir veri kümesi
    df = pd.DataFrame([
        {
            "id": 1111, "name": "Kahve Dünyası Beşiktaş", "latitude": 41.043, "longitude": 29.00,
            "tripadvisor_rating": 4.4, "tripadvisor_review_count": 320, "address": "Beşiktaş"
        },
        {
            "id": 2222, "name": "Hafız Mustafa Tatlı", "latitude": 41.033, "longitude": 28.99,
            "tripadvisor_rating": 4.6, "tripadvisor_review_count": 540, "address": "Kabataş"
        },
        {
            "id": 3333, "name": "Kolcuoğlu Ocakbaşı", "latitude": 41.056, "longitude": 28.98,
            "tripadvisor_rating": 4.5, "tripadvisor_review_count": 410, "address": "Bomonti"
        },
    ])
    df.to_excel(XLSX, index=False)

def test_00_health_and_reload():
    _ensure_excel_with_rows()
    with httpx.Client(timeout=10.0) as c:
        r = c.get(f"{BASE}/health")
        assert r.status_code == 200
        assert r.json().get("ok") is True

        # reload
        r = c.post(f"{BASE}/reload", json={})  # varsayılan /app/data/restaurants.xlsx
        assert r.status_code == 200
        j = r.json()
        assert j.get("ok") is True
        assert j.get("rows") >= 3

def test_01_candidates_basic():
    with httpx.Client(timeout=10.0) as c:
        r = c.get(f"{BASE}/candidates", params={"lat": 41.04, "lng": 29.02, "radius_km": 5, "limit": 5})
        assert r.status_code == 200
        j = r.json()
        assert "items" in j and j["count"] == len(j["items"])
        # Her kayıtta temel alanlar olmalı
        assert all("id" in it and "lat" in it and "lng" in it for it in j["items"])

def test_02_rank_no_prefs_returns_general_top():
    with httpx.Client(timeout=10.0) as c:
        r = c.get(f"{BASE}/rank", params={"lat": 41.04, "lng": 29.02, "radius_km": 5, "topK": 3})
        assert r.status_code == 200
        j = r.json()
        assert "items" in j and len(j["items"]) <= 3
        # components + score var mı?
        it = j["items"][0]
        assert "components" in it and "score" in it

def test_03_prefs_strict_filter_only_matching_categories():
    with httpx.Client(timeout=10.0) as c:
        # User tercihini coffee + dessert yap
        r = c.post(f"{BASE}/prefs/set", json={"userId": "mertay", "categories": ["coffee", "dessert"]})
        assert r.status_code == 200

        # Rank çağır → sadece coffee/dessert içeren kayıtlar gelmeli
        r = c.get(f"{BASE}/rank", params={
            "userId": "mertay", "lat": 41.04, "lng": 29.02, "radius_km": 5, "topK": 5
        })
        assert r.status_code == 200
        j = r.json()
        for it in j["items"]:
            cats = (it.get("categories") or "").lower()
            assert ("coffee" in cats) or ("dessert" in cats) or ("tatl" in cats) or ("kahve" in cats)

def test_04_prefs_no_match_returns_empty_list():
    with httpx.Client(timeout=10.0) as c:
        # User tercihini uç bir kategoriye çekelim
        r = c.post(f"{BASE}/prefs/set", json={"userId": "mertay", "categories": ["sushi-only-nonexistent"]})
        assert r.status_code == 200

        r = c.get(f"{BASE}/rank", params={
            "userId": "mertay", "lat": 41.04, "lng": 29.02, "radius_km": 2, "topK": 5
        })
        assert r.status_code == 200
        j = r.json()
        # Katı filtre nedeniyle, eşleşme yoksa boş dönmeli
        assert j.get("count", 0) == 0
        assert j.get("items") == []
