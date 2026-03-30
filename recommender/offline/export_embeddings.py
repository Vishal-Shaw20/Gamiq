import os
import psycopg2
import numpy as np
import pickle
from dotenv import load_dotenv

EMB_DIM = 1024    # changed from 384 — bge-large-en-v1.5 is 1024 dims
BATCH   = 50000

# ---------------- DB CONNECT ----------------

load_dotenv()
conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT')
)
cur = conn.cursor()

# -------- GET ACTUAL ROW COUNT --------

cur.execute("SELECT COUNT(*) FROM content_embeddings;")
TOTAL_ROWS = cur.fetchone()[0]

print(f"Total embeddings in DB: {TOTAL_ROWS}")

# -------- CREATE MEMMAP FILES --------

embeddings = np.memmap(
    "../artifacts/embeddings.memmap",
    dtype="float32",
    mode="w+",
    shape=(TOTAL_ROWS, EMB_DIM)
)

ids = np.memmap(
    "../artifacts/ids.memmap",
    dtype="int64",    # changed from int32 — game IDs may exceed int32 range
    mode="w+",
    shape=(TOTAL_ROWS,)
)

titles = []

# -------- STREAM BY PRIMARY KEY --------

last_id = 0
written = 0

print("Starting export...")

while True:

    cur.execute("""
                SELECT ce.game_id,
                       g.name,
                       ce.embedding
                FROM content_embeddings ce
                         JOIN games g ON g.id = ce.game_id
                WHERE ce.game_id > %s
                ORDER BY ce.game_id
                    LIMIT %s;
                """, (last_id, BATCH))

    rows = cur.fetchall()

    if not rows:
        break

    for game_id, title, emb in rows:
        embeddings[written] = np.array(emb, dtype="float32")
        ids[written]        = game_id
        titles.append(title)
        written += 1

    last_id = rows[-1][0]

    print(f"Written: {written}/{TOTAL_ROWS}")

# -------- FLUSH TO DISK --------

embeddings.flush()
ids.flush()

with open("../artifacts/titles.pkl", "wb") as f:
    pickle.dump(titles, f)

print("DONE — Export complete.")
conn.close()