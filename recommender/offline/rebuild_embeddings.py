import os
import psycopg2
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_values
from tqdm import tqdm
from datetime import datetime, UTC
import torch

torch.set_num_threads(os.cpu_count())

# ---------------- CONFIG ----------------

MODEL_NAME = "BAAI/bge-large-en-v1.5"

load_dotenv()

FETCH_BATCH = 2000
ENCODE_BATCH = 64
INSERT_BATCH = 2000

# RANGE CONTROL
START_ID = 0
END_ID = 1100000   # change this per run

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

# ---------------- TEXT BUILD ----------------

def clean_field(field):
    if isinstance(field, list):
        return ", ".join(field)
    if isinstance(field, dict):
        return field.get("name", "")
    return field or ""

def build_structured_text(row):
    name = row[1] or ""
    genres = clean_field(row[2])
    tags = clean_field(row[3])
    esrb = clean_field(row[4])
    developers = clean_field(row[5])
    publishers = clean_field(row[6])
    description = (row[7] or "")[:800]

    text = (
        f"CORE IDENTITY: This game is {name}, an {genres} title. "
        f"GAMEPLAY MECHANICS: {genres} gameplay involving {tags}. "
        f"NARRATIVE THEME: The setting and story involve {description}. "
        f"AUDIENCE: Rated {esrb} by ESRB. "
        f"STUDIO: Developed by {developers}, published by {publishers}."
    )

    return f"Represent this game for retrieving similar gameplay experiences: {text}"

# ---------------- MAIN ----------------

def main():

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to database...")
    read_conn = psycopg2.connect(**DB_CONFIG)
    write_conn = psycopg2.connect(**DB_CONFIG)

    write_cur = write_conn.cursor()
    write_cur.execute("SET synchronous_commit TO OFF;")

    # Count rows in this range
    count_cur = read_conn.cursor()
    count_cur.execute(
        "SELECT COUNT(*) FROM games WHERE id > %s AND id <= %s;",
        (START_ID, END_ID)
    )
    total_rows = count_cur.fetchone()[0]
    count_cur.close()

    print(f"Rows to process: {total_rows}")

    progress = tqdm(total=total_rows, desc="Processing")

    processed = 0
    last_id = START_ID
    inserted_total = 0

    while True:

        read_cur = read_conn.cursor()

        read_cur.execute("""
                         SELECT id, name, genres, tags,
                                esrb_rating, developers, publishers, description
                         FROM games
                         WHERE id > %s AND id <= %s
                         ORDER BY id
                             LIMIT %s
                         """, (last_id, END_ID, FETCH_BATCH))

        rows = read_cur.fetchall()
        read_cur.close()

        if not rows:
            break

        ids = [row[0] for row in rows]
        texts = [build_structured_text(row) for row in rows]

        embeddings = model.encode(
            texts,
            batch_size=ENCODE_BATCH,
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype("float32")

        now = datetime.now(UTC)

        insert_data = [
            (gid, emb.tolist(), now)
            for gid, emb in zip(ids, embeddings)
        ]

        execute_values(
            write_cur,
            """
            INSERT INTO content_embeddings (game_id, embedding, updated_at)
            VALUES %s
            """,
            insert_data,
            page_size=INSERT_BATCH
        )

        inserted_total += len(insert_data)

        last_id = rows[-1][0]

        processed += len(rows)
        progress.update(len(rows))

        # if processed % 10000 == 0:
        #     write_conn.commit()

        write_conn.commit()

    progress.close()

    write_conn.commit()

    write_cur.close()
    read_conn.close()
    write_conn.close()

    print("\n==============================")
    print("Embedding rebuild finished")
    print(f"Rows inserted in this run: {inserted_total}")
    print(f"Last processed id: {last_id}")
    print("==============================")

if __name__ == "__main__":
    main()