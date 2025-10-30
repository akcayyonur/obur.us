from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import math

app = FastAPI()

class Candidate(BaseModel):
    id: int
    name: str
    lat: float
    lng: float
    categories: Optional[str] = ""
    rating_avg: Optional[float] = None
    rating_count: Optional[int] = None

class RankRequest(BaseModel):
    userId: Optional[int] = None
    lat: float
    lng: float
    candidates: List[Candidate]

def score(user_prefs, p: Candidate, user_loc):
    rating = p.rating_avg if p.rating_avg is not None else 3.5
    votes = p.rating_count if p.rating_count is not None else 1
    cat_bonus = 0.3 if any(c.strip().lower() in user_prefs for c in (p.categories or '').split(',')) else 0.0
    d = math.dist([user_loc['lat'], user_loc['lng']], [p.lat, p.lng]) + 1e-6
    return (rating + math.log10(max(votes,1)) + cat_bonus) / d

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/rank")
def rank(req: RankRequest):
    user_prefs = {"kebap","coffee"}  # şimdilik sabit
    user_loc = {"lat": req.lat, "lng": req.lng}
    ranked = sorted(req.candidates, key=lambda p: -score(user_prefs, p, user_loc))
    return {"items": ranked[:50], "algo": "hybrid_stub_v1"}
