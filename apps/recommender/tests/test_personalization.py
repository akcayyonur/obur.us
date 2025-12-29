from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_user_recommendations():
    response = client.get("/users/1/recommendations?lat=41.0&lng=29.0")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert "items" in data
    assert len(data["items"]) > 0

def test_user_recommendations_with_price():
    response = client.get("/users/1/recommendations?lat=41.0&lng=29.0&price_range=2")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert "items" in data
    assert len(data["items"]) > 0
    for item in data["items"]:
        assert item["price_range"] is not None

def test_user_recommendations_not_found():
    response = client.get("/users/999/recommendations?lat=41.0&lng=29.0")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 999
    assert "items" in data
    assert len(data["items"]) == 0
