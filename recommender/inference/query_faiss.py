import faiss
import requests
import os
import psycopg2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from psycopg2.pool import SimpleConnectionPool
from concurrent.futures import ThreadPoolExecutor
import math

from recommender.inference.reranker import rerank

# -------------------- ENV --------------------

load_dotenv()

API_KEY = os.getenv("RAWG_API_KEY")

DB_CONFIG = {
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT"),
}

# -------------------- DB CONNECTION POOL --------------------

db_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    **DB_CONFIG
)

# -------------------- FAISS --------------------

BASE_DIR      = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

index = faiss.read_index(str(ARTIFACTS_DIR / "faiss_index.ivf"))

try:
    ivf_index        = faiss.downcast_index(index.index)
    ivf_index.nprobe = 256
except Exception as e:
    print(f"Warning: could not set nprobe: {e}")


# -------------------- TEXT BUILD --------------------

def clean_field(field):
    if isinstance(field, list):
        return ", ".join(field)
    if isinstance(field, dict):
        return field.get("name", "")
    return field or ""

def build_structured_text(row):
    name        = row[0] or ""
    genres      = clean_field(row[1])
    tags        = clean_field(row[2])
    esrb        = clean_field(row[3])
    developers  = clean_field(row[4])
    publishers  = clean_field(row[5])
    description = (row[6] or "")[:800]

    text = (
        f"CORE IDENTITY: This game is {name}, an {genres} title. "
        f"GAMEPLAY MECHANICS: {genres} gameplay involving {tags}. "
        f"NARRATIVE THEME: The setting and story involve {description}. "
        f"AUDIENCE: Rated {esrb} by ESRB. "
        f"STUDIO: Developed by {developers}, published by {publishers}."
    )

    return f"Represent this game for retrieving similar gameplay experiences: {text}"


# -------------------- DB FETCH FUNCTIONS --------------------

def fetch_embedding_from_db(game_id: int):
    conn = db_pool.getconn()
    cur  = conn.cursor()
    cur.execute(
        "SELECT embedding FROM content_embeddings WHERE game_id = %s;",
        (game_id,)
    )
    row = cur.fetchone()
    cur.close()
    db_pool.putconn(conn)

    if not row:
        return None

    emb = np.array(row[0], dtype="float32").reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb


def fetch_game_text(game_id: int) -> str | None:
    conn = db_pool.getconn()
    cur  = conn.cursor()
    cur.execute("""
                SELECT name, genres, tags, esrb_rating,
                       developers, publishers, description
                FROM games
                WHERE id = %s;
                """, (game_id,))
    row = cur.fetchone()
    cur.close()
    db_pool.putconn(conn)

    if not row:
        return None
    return build_structured_text(row)


def fetch_candidate_texts(game_ids: list[int]) -> dict[int, str]:
    conn = db_pool.getconn()
    cur  = conn.cursor()
    cur.execute("""
                SELECT id, name, genres, tags, esrb_rating,
                       developers, publishers, description
                FROM games
                WHERE id = ANY(%s);
                """, (game_ids,))
    rows = cur.fetchall()
    cur.close()
    db_pool.putconn(conn)

    result = {}
    for row in rows:
        result[row[0]] = build_structured_text(row[1:])
    return result


def fetch_developer_map(game_ids: list[int]) -> dict[int, str]:
    conn = db_pool.getconn()
    cur  = conn.cursor()
    cur.execute("""
                SELECT id, developers
                FROM games
                WHERE id = ANY(%s);
                """, (game_ids,))
    rows = cur.fetchall()
    cur.close()
    db_pool.putconn(conn)

    result = {}
    for game_id, developers in rows:
        if isinstance(developers, list) and developers:
            dev = developers[0].strip().lower()
        elif isinstance(developers, str) and developers:
            dev = developers.strip().lower()
        else:
            dev = "unknown"

        # Normalize to first word — groups studio subsidiaries automatically
        # e.g. "Rockstar North" → "rockstar", "Ubisoft Montreal" → "ubisoft"
        dev = dev.split()[0] if dev != "unknown" else "unknown"
        result[game_id] = dev

    return result


# -------------------- RAWG SERIES --------------------

def fetch_series_ids(game_id: int) -> set:
    series_ids = set()
    url = f"https://api.rawg.io/api/games/{game_id}/game-series?key={API_KEY}"

    while url:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                break
            data = response.json()
            for g in data.get("results", []):
                series_ids.add(g["id"])
            url = data.get("next")
        except Exception:
            break

    return series_ids


# -------------------- RECOMMENDATION --------------------

def get_recommendations(game_id: int, k: int = 10, max_per_series: int = 3):

    # ---- STAGE 1: FAISS retrieval ----

    query = fetch_embedding_from_db(game_id)
    if query is None:
        return []

    scores, returned_ids = index.search(query, 500)

    candidate_ids = [int(x) for x in returned_ids[0] if int(x) != game_id]

    # ---- QUALITY FILTER ----

    conn = db_pool.getconn()
    cur  = conn.cursor()
    cur.execute("""
                SELECT id FROM games
                WHERE id = ANY(%s)
                  AND ratings_count > 5
                """, (candidate_ids,))
    quality_ids   = {row[0] for row in cur.fetchall()}
    cur.close()
    db_pool.putconn(conn)

    candidate_ids = [cid for cid in candidate_ids if cid in quality_ids]
    candidate_ids = candidate_ids[:100]  # cap to keep reranker time consistent

    if not candidate_ids:
        return []

    # ---- STAGE 2: RERANK + SERIES FETCH IN PARALLEL ----

    query_text = fetch_game_text(game_id)
    if query_text is None:
        return []

    candidate_texts = fetch_candidate_texts(candidate_ids)

    candidates = [
        {"game_id": cid, "text": candidate_texts[cid]}
        for cid in candidate_ids
        if cid in candidate_texts
    ]

    # Run reranker and RAWG series fetch simultaneously
    with ThreadPoolExecutor(max_workers=2) as executor:
        rerank_future = executor.submit(rerank, query_text, candidates, 50)
        series_future = executor.submit(fetch_series_ids, game_id)

        reranked = rerank_future.result()

        try:
            series_ids = series_future.result(timeout=10)
        except Exception:
            series_ids = set()  # RAWG failed — skip series filter gracefully

    reranked_ids = [c["game_id"] for c in reranked]

    # ---- QUALITY SCORING ----

    conn = db_pool.getconn()
    cur  = conn.cursor()
    cur.execute("""
                SELECT id, rating, ratings_count, metacritic
                FROM games
                WHERE id = ANY(%s)
                """, (reranked_ids,))
    rows = cur.fetchall()
    cur.close()
    db_pool.putconn(conn)

    meta            = {r[0]: r[1:] for r in rows}
    reranker_scores = {c["game_id"]: c["reranker_score"] for c in reranked}
    faiss_scores    = {
        int(cid): float(score)
        for cid, score in zip(returned_ids[0], scores[0])
        if int(cid) != game_id
    }

    ranked = []

    for cand_id in reranked_ids:
        if cand_id not in meta:
            continue

        rating, count, meta_score = meta[cand_id]

        rating     = rating     or 0
        count      = count      or 0
        meta_score = meta_score or 0

        rating_norm    = rating / 5
        meta_norm      = meta_score / 100
        count_score    = min(math.log10(count + 1) / math.log10(1_000_000), 1.0)
        reranker_score = reranker_scores.get(cand_id, 0.0)

        final_score = (
                0.5 * reranker_score +
                0.3 * faiss_scores.get(cand_id, 0.0) +
                0.1 * rating_norm +
                0.05 * meta_norm +
                0.05 * count_score
        )

        ranked.append((cand_id, final_score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    ranked_ids = [x[0] for x in ranked]

    # ---- SERIES FILTER + DEVELOPER CAP ----

    dev_map = fetch_developer_map(ranked_ids)

    result       = []
    seen         = set()
    series_taken = 0
    dev_counts   = {}
    max_per_dev  = 2

    for cand_id in ranked_ids:

        if cand_id in seen:
            continue

        if cand_id in series_ids:
            if series_taken >= max_per_series:
                continue
            series_taken += 1

        dev = dev_map.get(cand_id, "unknown")
        if dev != "unknown":
            if dev_counts.get(dev, 0) >= max_per_dev:
                continue
            dev_counts[dev] = dev_counts.get(dev, 0) + 1

        result.append(cand_id)
        seen.add(cand_id)

        if len(result) == k:
            break

    return result