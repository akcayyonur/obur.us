# apps/recommender/main.py
from fastapi import FastAPI, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import math
import os
import time
import hashlib

app = FastAPI(title="OBURUS Recommender", version="1.5.0")

# -----------------------------
# Config (ENV ile ayarlanabilir)
# -----------------------------
DATA_XLSX = os.getenv("DATA_XLSX", "/app/data/restaurants.xlsx")  # Excel yolu
W_RATING = float(os.getenv("W_RATING", "0.45"))
W_POP    = float(os.getenv("W_POP", "0.15"))
W_CAT    = float(os.getenv("W_CAT", "0.25"))
W_DIST   = float(os.getenv("W_DIST", "0.15"))
MAX_KM   = float(os.getenv("MAX_KM",  "20"))
TOP_K    = int(os.getenv("TOP_K", "50"))

DEFAULT_PREFS = {"kebap", "coffee", "burger"}
USER_PREFS: Dict[str, set] = {}

FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "/app/feedback.jsonl")

# Bellekte aday mağazası
STORE_COLS = ["id", "name", "lat", "lng", "categories", "rating_avg", "rating_count", "address", "source"]
STORE: pd.DataFrame = pd.DataFrame(columns=STORE_COLS)

# -----------------------------
# Yardımcılar (JSON/NaN/∞ güvenliği + coğrafya)
# -----------------------------
def _is_finite_number(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def _json_sanitize_value(v):
    # numpy tiplerini primitive'e çevir
    if isinstance(v, (np.floating, np.float32, np.float64)):
        v = float(v)
    if isinstance(v, (np.integer, np.int32, np.int64)):
        v = int(v)
    # NaN/±inf -> None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v

def _json_sanitize_record(d: dict) -> dict:
    return {k: _json_sanitize_value(v) for k, v in d.items()}

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2)**2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlmb/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def bayesian_rating(avg: Optional[float], n: Optional[int], C: float = 3.8, m: int = 50) -> float:
    R = avg if (avg is not None and not (isinstance(avg, float) and math.isnan(avg))) else C
    v = n if (n is not None and n > 0) else 0
    return (v / (v + m)) * R + (m / (v + m)) * C

def pop_score(n: Optional[int], cap: int = 1000) -> float:
    v = max(0, int(n) if n is not None else 0)
    return min(1.0, math.log10(v + 1) / math.log10(cap + 1))

def cat_score_from_match(matched_cnt: int, prefs_cnt: int) -> float:
    if prefs_cnt <= 0:
        return 0.0
    return matched_cnt / float(prefs_cnt)

def dist_score_km(km: float, max_km: float = MAX_KM) -> float:
    if km <= 0:
        return 1.0
    if km >= max_km:
        return 0.0
    return 1.0 - (km / max_km)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def get_user_prefs(user_id: Optional[str]) -> set:
    return USER_PREFS.get(user_id, set())

def _stable_id(name: str, lat: float, lng: float) -> int:
    """ID boşsa deterministik fallback ID üretir."""
    s = f"{name}|{lat:.6f}|{lng:.6f}"
    return int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16) & 0x7FFFFFFFFFFFFFFF

# -----------------------------
# İsimden kategori çıkarımı (Excel'de categories boşsa)
# -----------------------------
def infer_categories_from_name(name: str) -> str:
    txt = (name or "").lower()
    tags = set()
    rules = {
        "coffee": ["coffee","kahve","cafe","café"],
        "dessert": ["dessert","tatl","pastane","patisserie","baklava","kunefe","künefe","muhallebi"],
        "kebap": ["kebap","ocakba","ocakbaşı","adana","urfa"],
        "burger": ["burger"],
        "fish": ["balık","fish","seafood","meyhane"],
        "breakfast": ["kahvalt"],
        "pizza": ["pizza","pizzeria"]
    }
    for cat, keys in rules.items():
        if any(k in txt for k in keys):
            tags.add(cat)
    return ",".join(sorted(tags))

# -----------------------------
# Excel yükleme / yenileme
# -----------------------------
def load_from_excel(path: str) -> pd.DataFrame:
    """
    Beklenen eşlemeler:
    - id            <- restaurant_id | id (boşsa stable hash)
    - name          <- name
    - lat/lng       <- latitude / longitude
    - rating_avg    <- tripadvisor_rating | rating | rating_avg
    - rating_count  <- tripadvisor_review_count | review_count | rating_count
    - address       <- address | (street1 + city)
    - categories    <- (Excel'de yoksa isimden çıkarım)
    - source        <- 'excel'
    """
    df = pd.read_excel(path)
    cols = {c.lower(): c for c in df.columns}

    def pick(*cands):
        for key in cands:
            if key in cols:
                return cols[key]
        return None

    c_id   = pick("restaurant_id", "id")
    c_name = pick("name")
    c_lat  = pick("latitude", "lat", "y")
    c_lng  = pick("longitude", "lng", "x")
    c_addr = pick("address")
    c_rat  = pick("tripadvisor_rating", "rating", "rating_avg")
    c_cnt  = pick("tripadvisor_review_count", "review_count", "rating_count")

    if not all([c_name, c_lat, c_lng]):
        raise ValueError("Excel'de zorunlu sütunlar yok: name/latitude/longitude (+ restaurant_id önerilir).")

    # Adres fallback’i
    if not c_addr:
        street1 = pick("street1"); city = pick("city")
        if street1 and city:
            df["__address"] = df[street1].astype(str).str.strip() + ", " + df[city].astype(str).str.strip()
            c_addr = "__address"
        else:
            df["__address"] = None
            c_addr = "__address"

    # Tip zorlama
    df["__name"] = df[c_name].astype(str)
    df["__lat"]  = pd.to_numeric(df[c_lat], errors="coerce")
    df["__lng"]  = pd.to_numeric(df[c_lng], errors="coerce")
    if c_id:
        df["__id"] = pd.to_numeric(df[c_id], errors="coerce")
    else:
        df["__id"] = np.nan

    # Eksikleri temizle/üret: name/lat/lng boş olanları at
    df = df.dropna(subset=["__name", "__lat", "__lng"])

    # Boş ID için deterministik fallback
    mask = df["__id"].isna()
    if mask.any():
        df.loc[mask, "__id"] = df.loc[mask].apply(
            lambda r: _stable_id(str(r["__name"]), float(r["__lat"]), float(r["__lng"])),
            axis=1
        )

    # rating/review_count
    rat = pd.to_numeric(df[c_rat], errors="coerce") if c_rat else pd.Series([np.nan]*len(df))
    cnt = pd.to_numeric(df[c_cnt], errors="coerce") if c_cnt else pd.Series([0]*len(df))

    out = pd.DataFrame({
        "id":           df["__id"].astype("int64"),
        "name":         df["__name"].astype(str),
        "lat":          df["__lat"].astype(float),
        "lng":          df["__lng"].astype(float),
        "categories":   "",  # önce boş, sonra isimden dolduracağız
        "rating_avg":   rat,
        "rating_count": cnt.fillna(0).astype(int),
        "address":      df[c_addr],  # None kalabilir, JSON'da null döner
        "source":       "excel",
    })

    # isimden kategori çıkarımı (boş olanlar için)
    empty_mask = (out["categories"].isna()) | (out["categories"].astype(str).str.strip() == "")
    out.loc[empty_mask, "categories"] = out.loc[empty_mask, "name"].apply(infer_categories_from_name)

    # sonsuzluk/NaN temizliği ve lat/lng'nin finite olduğundan emin ol
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["id", "lat", "lng"])
    out = out[out["lat"].apply(_is_finite_number) & out["lng"].apply(_is_finite_number)]
    out["id"] = out["id"].astype("int64")
    return out[STORE_COLS]

def ensure_store_loaded():
    global STORE
    if STORE.empty:
        STORE = load_from_excel(DATA_XLSX)

# -----------------------------
# Modeller
# -----------------------------
class Candidate(BaseModel):
    id: int
    name: str
    lat: float
    lng: float
    categories: Optional[str] = ""
    rating_avg: Optional[float] = None
    rating_count: Optional[int] = None
    address: Optional[str] = None
    source: Optional[str] = None

class RankRequest(BaseModel):
    userId: Optional[str] = Field(default=None)
    lat: float
    lng: float
    candidates: Optional[List[Candidate]] = None
    topK: Optional[int] = None
    radius_km: Optional[float] = Field(default=None, description="Kandidatsız modda Excel'den çekme yarıçapı")

class RankResponse(BaseModel):
    items: List[Dict[str, Any]]
    algo: str = "hybrid_v1_excel"
    weights: Dict[str, float]
    explainable: bool = True
    count: int

class FeedbackEvent(BaseModel):
    userId: Optional[str] = None
    action: str
    placeId: int
    context: Dict[str, Any] = {}
    ts: float = Field(default_factory=lambda: time.time())

class PrefsPayload(BaseModel):
    userId: str
    categories: List[str]

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "algo": "hybrid_v1_excel"}

@app.post("/reload")
def reload_store(path: Optional[str] = Body(default=None, embed=True)):
    """Excel’i yeniden yükler. path verilmezse DATA_XLSX kullanılır."""
    global STORE
    p = path or DATA_XLSX
    try:
        STORE = load_from_excel(p)
        return {"ok": True, "rows": int(len(STORE)), "path": p}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yükleme hatası: {e}")

@app.get("/debug/weights")
def debug_weights():
    return {
        "rating": W_RATING, "popularity": W_POP, "category": W_CAT, "distance": W_DIST,
        "max_km": MAX_KM, "top_k": TOP_K,
        "data_xlsx": DATA_XLSX,
        "store_rows": int(len(STORE)) if not STORE.empty else 0
    }

@app.get("/candidates")
def list_candidates(lat: float, lng: float, radius_km: float = 20.0, limit: int = 200):
    """Excel mağazasından konuma en yakın adayları getirir (skorlamasız)."""
    ensure_store_loaded()
    df = STORE.copy()

    # Geçerli lat/lng
    df = df[df["lat"].apply(_is_finite_number) & df["lng"].apply(_is_finite_number)].copy()

    # Mesafe
    df["km"] = df.apply(lambda r: haversine_km(float(lat), float(lng), float(r["lat"]), float(r["lng"])), axis=1)

    # Filtre + sırala + limit
    df = df[df["km"].apply(_is_finite_number) & (df["km"] <= float(radius_km))] \
           .sort_values("km", ascending=True).head(int(limit))

    # JSON güvenliği
    records = []
    for _, r in df.iterrows():
        rec = {
            "id": int(r["id"]),
            "name": str(r["name"]),
            "lat": float(r["lat"]),
            "lng": float(r["lng"]),
            "categories": ("" if pd.isna(r.get("categories")) else str(r.get("categories"))),
            "rating_avg": (None if pd.isna(r.get("rating_avg")) else float(r.get("rating_avg"))),
            "rating_count": (None if pd.isna(r.get("rating_count")) else int(r.get("rating_count"))),
            "address": (None if pd.isna(r.get("address")) else str(r.get("address"))),
            "source": (None if pd.isna(r.get("source")) else str(r.get("source"))),
            "km": float(r["km"]),
        }
        records.append(_json_sanitize_record(rec))

    payload = {"count": len(records), "items": records}
    return JSONResponse(content=jsonable_encoder(payload), status_code=200)

@app.post("/prefs/set")
def set_prefs(payload: PrefsPayload):
    cats = {c.strip().lower() for c in payload.categories if c and c.strip()}
    USER_PREFS[payload.userId] = cats if cats else set()
    return {"ok": True, "userId": payload.userId, "prefs": sorted(USER_PREFS[payload.userId])}

@app.post("/feedback")
def feedback(ev: FeedbackEvent):
    try:
        line = {"ts": ev.ts, "userId": ev.userId, "action": ev.action, "placeId": ev.placeId, "context": ev.context}
        with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
            f.write(f"{line}\n")
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"feedback yazılamadı: {e}")

# -----------------------------
# Rank (POST)
# -----------------------------
@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest):
    try:
        prefs_set = get_user_prefs(req.userId)  # boş olabilir
        top_k = req.topK if (req.topK and req.topK > 0) else TOP_K

        # 1) Kandidatsız mod: Excel'den çek
        if not req.candidates:
            ensure_store_loaded()
            radius = float(req.radius_km or MAX_KM)

            df = STORE.copy()
            df = df[df["lat"].apply(_is_finite_number) & df["lng"].apply(_is_finite_number)].copy()
            df["km"] = df.apply(lambda r: haversine_km(req.lat, req.lng, float(r["lat"]), float(r["lng"])), axis=1)

            df = df[df["km"].apply(_is_finite_number) & (df["km"] <= radius)] \
                   .sort_values("km", ascending=True).head(1000)  # genişletilmiş aday havuzu

            df = df.dropna(subset=["id", "lat", "lng", "km"])
            df["id"] = df["id"].astype("int64")

            # --- KATI PREFS FİLTRESİ ---
            if prefs_set:
                def _tokens(cats: Optional[str]) -> set:
                    return {t.strip().lower() for t in (cats or "").split(",") if t.strip()}
                df = df[df["categories"].apply(lambda c: len(_tokens(c) & prefs_set) > 0)]
                if df.empty:
                    payload = {
                        "items": [],
                        "algo": "hybrid_v1_excel",
                        "weights": {"rating": W_RATING, "popularity": W_POP, "category": W_CAT, "distance": W_DIST},
                        "explainable": True,
                        "count": 0,
                        "note": f"Tercihlerle eşleşen sonuç yok: {sorted(prefs_set)}"
                    }
                    return JSONResponse(content=jsonable_encoder(payload), status_code=200)
            # --- /KATI PREFS FİLTRESİ ---

            candidates: List[Candidate] = []
            for _, r in df.iterrows():
                candidates.append(Candidate(
                    id=int(r["id"]),
                    name=str(r["name"]),
                    lat=float(r["lat"]),
                    lng=float(r["lng"]),
                    categories=str(r.get("categories") or ""),
                    rating_avg=None if pd.isna(r.get("rating_avg")) else float(r.get("rating_avg")),
                    rating_count=None if pd.isna(r.get("rating_count")) else int(r.get("rating_count")),
                    address=None if pd.isna(r.get("address")) else str(r.get("address")),
                    source=None if pd.isna(r.get("source")) else str(r.get("source")),
                ))
        else:
            # dışarıdan aday listesi geldiyse
            candidates = req.candidates
            # --- KATI PREFS FİLTRESİ (gelen adaylar üzerinde) ---
            if prefs_set:
                def _tokens(cats: Optional[str]) -> set:
                    return {t.strip().lower() for t in (cats or "").split(",") if t.strip()}
                filtered = [c for c in candidates if len(_tokens(c.categories) & prefs_set) > 0]
                candidates = filtered
                if not candidates:
                    payload = {
                        "items": [],
                        "algo": "hybrid_v1_excel",
                        "weights": {"rating": W_RATING, "popularity": W_POP, "category": W_CAT, "distance": W_DIST},
                        "explainable": True,
                        "count": 0,
                        "note": f"Tercihlerle eşleşen sonuç yok: {sorted(prefs_set)}"
                    }
                    return JSONResponse(content=jsonable_encoder(payload), status_code=200)
            # --- /KATI PREFS FİLTRESİ ---

        # 2) Skorlama
        items: List[Dict[str, Any]] = []
        for c in candidates:
            km = haversine_km(req.lat, req.lng, c.lat, c.lng)
            r_safe = bayesian_rating(c.rating_avg, c.rating_count) / 5.0
            p_sc   = pop_score(c.rating_count)
            tokens = {t.strip().lower() for t in (c.categories or "").split(",") if t.strip()}
            matched = sorted(tokens.intersection(prefs_set))
            c_sc   = cat_score_from_match(len(matched), len(prefs_set))
            d_sc   = dist_score_km(km)
            score  = clamp01(W_RATING*r_safe + W_POP*p_sc + W_CAT*c_sc + W_DIST*d_sc)

            items.append({
                "id": int(c.id),
                "name": str(c.name),
                "lat": float(c.lat),
                "lng": float(c.lng),
                "address": None if (c.address is None) else str(c.address),
                "source": None if (c.source is None) else str(c.source),
                "categories": (c.categories or ""),
                "matched_prefs": matched,
                "score": float(round(score, 6)),
                "components": {
                    "rating": float(round(r_safe, 6)),
                    "popularity": float(round(p_sc, 6)),
                    "category": float(round(c_sc, 6)),
                    "distance": float(round(d_sc, 6)),
                    "km": float(round(km, 3)),
                    "cat_tokens": sorted(tokens)
                }
            })

        items.sort(key=lambda x: x["score"], reverse=True)
        items = items[:top_k]
        items = [_json_sanitize_record(it) for it in items]

        payload = {
            "items": items,
            "algo": "hybrid_v1_excel",
            "weights": {"rating": W_RATING, "popularity": W_POP, "category": W_CAT, "distance": W_DIST},
            "explainable": True,
            "count": len(items),
        }
        return JSONResponse(content=jsonable_encoder(payload), status_code=200)

    except Exception as e:
        import traceback, sys
        tb = "".join(traceback.format_exception(*sys.exc_info()))
        return JSONResponse(status_code=500, content={"detail": f"rank error: {e}", "trace": tb[:2000]})

# -----------------------------
# Tarayıcıdan GET ile /rank denemesi (POST /rank'i kullanır)
# -----------------------------
@app.get("/rank")
def rank_get(
    lat: float,
    lng: float,
    radius_km: float = 20.0,
    topK: int = 50,
    userId: Optional[str] = None
):
    """
    Tarayıcıdan deneme için GET sarmalayıcı.
    Örnek:
      /rank?lat=41.04&lng=29.02&radius_km=5&topK=5
    """
    req = RankRequest(
        userId=userId,
        lat=lat,
        lng=lng,
        radius_km=radius_km,
        topK=topK,
        candidates=None
    )
    return rank(req)  # mevcut POST /rank fonksiyonunu kullanır
