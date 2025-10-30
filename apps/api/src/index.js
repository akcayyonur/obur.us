const Fastify = require('fastify');
const cors = require('@fastify/cors');
const axios = require('axios');
const { Pool } = require('pg');

const fastify = Fastify({ logger: true });

// --- PG bağlantısı ---
const pool = new Pool({
  host: process.env.DB_HOST || 'db',
  port: Number(process.env.DB_PORT || 5432),
  database: process.env.DB_NAME || 'oburus',
  user: process.env.DB_USER || 'oburus',
  password: process.env.DB_PASS || 'oburus_pass',
  max: 10
});

// CORS
fastify.register(cors, { origin: true });

// Health
fastify.get('/health', async () => ({ ok: true }));

// Yakındaki yerler
fastify.get('/places', async (req, reply) => {
  try {
    const lat = Number(req.query.lat);
    const lng = Number(req.query.lng);
    const limit = Number(req.query.limit || 50);
    const radius = Number(req.query.radius || 1500); // metre (opsiyonel)

    if (Number.isNaN(lat) || Number.isNaN(lng)) {
      return reply.code(400).send({ error: 'lat,lng gerekli' });
    }

    const sql = `
      SELECT id, name, address, categories, rating_avg, rating_count, source,
             ST_Distance(
               location_geo,
               ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography
             ) AS distance_m,
             ST_Y(location_geo::geometry) AS lat,
             ST_X(location_geo::geometry) AS lng
      FROM places
      WHERE ST_DWithin(
        location_geo,
        ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography,
        $3
      )
      ORDER BY distance_m ASC
      LIMIT $4;
    `;
    const { rows } = await pool.query(sql, [lng, lat, radius, limit]);
    reply.send(rows);
  } catch (err) {
    fastify.log.error(err);
    reply.code(500).send({ error: 'places sorgusunda hata' });
  }
});

// Öneriler (recommender’a adaylar + konum gönder)
fastify.get('/recommendations', async (req, reply) => {
  try {
    const lat = Number(req.query.lat);
    const lng = Number(req.query.lng);
    const userId = req.query.userId || null;

    if (Number.isNaN(lat) || Number.isNaN(lng)) {
      return reply.code(400).send({ error: 'lat,lng gerekli' });
    }

    const candSql = `
      SELECT id, name, categories, rating_avg, rating_count,
             ST_Y(location_geo::geometry) AS lat,
             ST_X(location_geo::geometry) AS lng
      FROM places
      ORDER BY ST_Distance(
               location_geo,
               ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography
             ) ASC
      LIMIT 200;
    `;
    const { rows: candidates } = await pool.query(candSql, [lng, lat]);

    const res = await axios.post(`${process.env.RECO_URL || 'http://recommender:8000'}/rank`, {
      userId, lat, lng, candidates
    });

    reply.send(res.data);
  } catch (err) {
    fastify.log.error(err);
    reply.code(500).send({ error: 'recommendations sırasında hata' });
  }
});


fastify.listen({ host: '0.0.0.0', port: 3000 })
  .catch(err => { fastify.log.error(err); process.exit(1); });
