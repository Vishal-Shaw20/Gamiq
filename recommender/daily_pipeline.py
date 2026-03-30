import os
import json
import time
import requests
import psycopg2
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from pathlib import Path

# ============================================================
# -------------------- ENV CONFIG ----------------------------
# ============================================================

load_dotenv()

# API key rotation — fallback style
RAWG_API_KEYS = [
    os.getenv("RAWG_API_KEY"),
    os.getenv("RAWG_API_KEY_1"),
    os.getenv("RAWG_API_KEY_2"),
    os.getenv("RAWG_API_KEY_3"),
    os.getenv("RAWG_API_KEY_4"),
    os.getenv("RAWG_API_KEY_5"),
    os.getenv("RAWG_API_KEY_6"),
    os.getenv("RAWG_API_KEY_7"),
]
RAWG_API_KEYS    = [k for k in RAWG_API_KEYS if k]  # remove None if fewer than 8 set
current_key_index = 0

DB_CONFIG = {
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT"),
}

BASE_DIR         = Path(__file__).resolve().parent
ARTIFACTS_DIR    = BASE_DIR / "artifacts"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss_index.ivf"
CHECKPOINT_PATH  = ARTIFACTS_DIR / "checkpoint.txt"

RAWG_BASE = "https://api.rawg.io/api/games"
MAX_PAGES = 100

# Load model once — must match model used to build existing index
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# ============================================================
# -------------------- API KEY ROTATION ----------------------
# ============================================================

def rawg_get(url: str, timeout: int = 10):
    """Make RAWG GET request with automatic key rotation on 429."""
    global current_key_index

    for attempt in range(len(RAWG_API_KEYS)):
        key      = RAWG_API_KEYS[current_key_index]
        sep      = "&" if "?" in url else "?"
        full_url = f"{url}{sep}key={key}"

        try:
            response = requests.get(full_url, timeout=timeout)
        except Exception as e:
            print(f"Request error: {e}")
            return None

        if response.status_code == 429:
            print(f"Rate limit hit on key {current_key_index + 1}, switching...")
            current_key_index = (current_key_index + 1) % len(RAWG_API_KEYS)
            time.sleep(1)
            continue

        return response

    print("All API keys exhausted.")
    return None

# ============================================================
# -------------------- HELPERS -------------------------------
# ============================================================

def clean_text(x):
    return None if not x or str(x).strip() == "" else str(x)

def clean_int(x):
    return None if x is None else int(x)

def extract_names(field):
    if not field:
        return []
    return [item["name"] for item in field if "name" in item]

def extract_platforms(field):
    if not field:
        return []
    return [p["platform"]["name"] for p in field if "platform" in p]

# ============================================================
# -------------------- TEXT BUILD ----------------------------
# ============================================================
# Must be IDENTICAL to build_structured_text() in query_faiss.py
# New game embeddings must match existing index embedding space

def build_structured_text(g: dict) -> str:
    name        = clean_text(g.get("name")) or ""
    genres      = ", ".join(extract_names(g.get("genres")))
    tags        = ", ".join(extract_names(g.get("tags")))
    esrb        = g.get("esrb_rating", {}).get("name", "") if g.get("esrb_rating") else ""
    developers  = ", ".join(extract_names(g.get("developers")))
    publishers  = ", ".join(extract_names(g.get("publishers")))
    description = (g.get("description_raw") or "")[:800]

    text = (
        f"CORE IDENTITY: This game is {name}, an {genres} title. "
        f"GAMEPLAY MECHANICS: {genres} gameplay involving {tags}. "
        f"NARRATIVE THEME: The setting and story involve {description}. "
        f"AUDIENCE: Rated {esrb} by ESRB. "
        f"STUDIO: Developed by {developers}, published by {publishers}."
    )

    return f"Represent this game for retrieving similar gameplay experiences: {text}"

# ============================================================
# -------------------- CHECKPOINT ----------------------------
# ============================================================

def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            val = f.read().strip()
            return int(val) if val else None

    # No checkpoint file — use the highest game ID in DB as baseline
    print("No checkpoint file. Using MAX(id) from DB as baseline...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()
    cur.execute("SELECT COALESCE(MAX(id), 0) FROM games;")
    max_id = cur.fetchone()[0]
    cur.close()
    conn.close()

    save_checkpoint(max_id)
    print(f"Baseline set to MAX game ID in DB: {max_id}")
    return max_id

def save_checkpoint(game_id: int):
    with open(CHECKPOINT_PATH, "w") as f:
        f.write(str(game_id))
    print(f"Checkpoint saved: {game_id}")

# ============================================================
# -------------------- DB INSERTS ----------------------------
# ============================================================

def insert_games_batch(conn, rows):
    sql = """
          INSERT INTO games (
              id, slug, name, name_original,
              description, description_raw,
              released,
              background_image, background_image_additional,
              suggestions_count,
              platforms, developers, publishers, genres, tags,
              esrb_rating, website,
              screenshots_count, achievements_count, game_series_count, additions_count,
              parents_count, alternative_names,
              rating, ratings_count, metacritic
          )
          VALUES %s
              ON CONFLICT (id) DO NOTHING; \
          """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()


def insert_embeddings_batch(conn, rows):
    """
    rows: list of (game_id, embedding_list)
    embedding_list: list of 1024 floats — matches bge-large-en-v1.5
    """
    register_vector(conn)
    sql = """
          INSERT INTO content_embeddings (game_id, embedding)
          VALUES %s
              ON CONFLICT (game_id)
        DO UPDATE SET embedding   = EXCLUDED.embedding,
                             updated_at  = NOW(); \
          """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()

# ============================================================
# -------------------- RAWG FETCHING -------------------------
# ============================================================

def fetch_new_game_ids(checkpoint_id):
    page     = 1
    new_ids  = []
    first_id = None

    while page <= MAX_PAGES:
        print(f"Fetching RAWG page {page}")

        url      = f"{RAWG_BASE}?ordering=-created&page={page}"
        response = rawg_get(url)

        if response is None or response.status_code != 200:
            print("RAWG request failed")
            break

        data    = response.json()
        results = data.get("results", [])

        if not results:
            break

        for g in results:
            gid = g["id"]

            if first_id is None:
                first_id = gid

            if gid == checkpoint_id:
                print(f"Reached checkpoint ID {checkpoint_id}. Stop.")
                return new_ids, first_id

            new_ids.append(gid)

        page += 1

    if page > MAX_PAGES:
        print(f"Warning: hit MAX_PAGES ({MAX_PAGES}) limit.")

    return new_ids, first_id


def fetch_game_details(game_id):
    url      = f"{RAWG_BASE}/{game_id}"
    response = rawg_get(url)

    if response is None or response.status_code != 200:
        return None

    return response.json()

# ============================================================
# -------------------- FAISS UPDATE --------------------------
# ============================================================

def update_faiss(new_ids: list, new_vectors: list):
    """
    Appends new game vectors to existing FAISS index and npy files.
    Vectors are normalized before adding — matches train_faiss.py behavior.
    Note: vectors added after IVF training go to flat overflow.
    For small daily additions this is acceptable.
    Full rebuild should be triggered periodically via train_faiss.py.
    """

    if not FAISS_INDEX_PATH.exists():
        print("FAISS index missing — skipping FAISS update.")
        return

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    vectors = np.array(new_vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    ids = np.array(new_ids, dtype="int64")

    # IndexIDMap wraps inner IVFFlat — add_with_ids works correctly on IDMap
    index.add_with_ids(vectors, ids)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"FAISS updated — added {len(new_ids)} vectors.")

# ============================================================
# -------------------- MAIN DAILY PIPELINE -------------------
# ============================================================

def run_daily_pipeline():

    print("Starting daily pipeline...")

    conn = psycopg2.connect(**DB_CONFIG)

    checkpoint_id = load_checkpoint()
    print(f"Checkpoint ID: {checkpoint_id}")

    new_ids, first_id = fetch_new_game_ids(checkpoint_id)

    if not new_ids:
        print("No new games.")
        conn.close()
        return

    print(f"Found {len(new_ids)} new games")

    game_rows      = []
    embedding_rows = []
    faiss_vectors  = []
    faiss_ids      = []

    for gid in reversed(new_ids):  # oldest first

        g = fetch_game_details(gid)

        if not g:
            continue

        # -------- Game row --------
        game_rows.append((
            clean_int(g.get("id")),
            clean_text(g.get("slug")),
            clean_text(g.get("name")),
            clean_text(g.get("name_original")),
            clean_text(g.get("description")),
            clean_text(g.get("description_raw")),
            clean_text(g.get("released")),
            clean_text(g.get("background_image")),
            clean_text(g.get("background_image_additional")),
            clean_int(g.get("suggestions_count")),
            extract_platforms(g.get("platforms")),
            extract_names(g.get("developers")),
            extract_names(g.get("publishers")),
            extract_names(g.get("genres")),
            extract_names(g.get("tags")),
            json.dumps(g.get("esrb_rating")) if g.get("esrb_rating") else None,
            clean_text(g.get("website")),
            clean_int(g.get("screenshots_count")),
            clean_int(g.get("achievements_count")),
            clean_int(g.get("game_series_count")),
            clean_int(g.get("additions_count")),
            clean_int(g.get("parents_count")),
            extract_names(g.get("alternative_names")),
            g.get("rating") or 0.0,
            clean_int(g.get("ratings_count")),
            clean_int(g.get("metacritic")),
        ))

        # -------- Embedding --------
        # build_structured_text must match query_faiss.py exactly
        text      = build_structured_text(g)
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,  # normalization applied at FAISS stage
        ).astype("float32")

        embedding_rows.append((gid, embedding.tolist()))
        faiss_vectors.append(embedding)
        faiss_ids.append(gid)

        print(f"Prepared game {gid}: {g.get('name')}")
        time.sleep(0.2)

    # -------- Batch inserts --------
    if game_rows:
        insert_games_batch(conn, game_rows)
        print(f"Inserted {len(game_rows)} games to DB.")

    if embedding_rows:
        insert_embeddings_batch(conn, embedding_rows)
        print(f"Inserted {len(embedding_rows)} embeddings to DB.")

    # -------- FAISS update --------
    if faiss_vectors:
        update_faiss(faiss_ids, faiss_vectors)

    if first_id is not None:
        save_checkpoint(first_id)

    conn.close()
    print("Daily pipeline completed.")

# ============================================================

if __name__ == "__main__":
    run_daily_pipeline()