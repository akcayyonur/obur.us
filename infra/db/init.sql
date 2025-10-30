-- PostGIS uzansı
CREATE EXTENSION IF NOT EXISTS postgis;

-- Şema temizliği (opsiyonel; geliştirmede işine yarar)
DROP TABLE IF EXISTS reviews CASCADE;
DROP TABLE IF EXISTS places CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- users
CREATE TABLE users (
  id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  email            TEXT NOT NULL UNIQUE,
  hashed_password  BYTEA NOT NULL,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- places (MSSQL'deki location_geo GEOGRAPHY karşılığı)
-- PostGIS'te POINT'in koordinatı: POINT(lon lat)
CREATE TABLE places (
  id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  name          TEXT NOT NULL,
  location_geo  GEOGRAPHY(POINT,4326) NOT NULL,
  address       TEXT,
  categories    TEXT,
  rating_avg    DOUBLE PRECISION,
  rating_count  INTEGER,
  source        TEXT
);

-- Uzaysal indeks (MSSQL spatial index karşılığı)
CREATE INDEX IF NOT EXISTS ix_places_location_geo ON places USING GIST (location_geo);

-- reviews
CREATE TABLE reviews (
  id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  user_id    BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  place_id   BIGINT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
  rating     DOUBLE PRECISION NOT NULL,
  text       TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Küçük test verisi (3 restoran)
-- DİKKAT: PostGIS'te POINT(lon lat). Senin kayıtların (lat,lon) idi.
-- Beşiktaş (29.01E, 41.045N), Kadıköy (29.03E, 40.99N), Şişli (28.99E, 41.06N)
INSERT INTO places (name,address,categories,rating_avg,rating_count,location_geo,source) VALUES
('Test Kebap','Beşiktaş','kebap',4.6,320,ST_GeogFromText('SRID=4326;POINT(29.0100 41.0450)'),'seed'),
('Test Kahve','Kadıköy','coffee',4.3,210,ST_GeogFromText('SRID=4326;POINT(29.0300 40.9900)'),'seed'),
('Test Burger','Şişli','burger',4.1,150,ST_GeogFromText('SRID=4326;POINT(28.9900 41.0600)'),'seed');
