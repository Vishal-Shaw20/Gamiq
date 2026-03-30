import numpy as np
import pickle
import os

EMB_DIM = 1024    # changed from 384 — bge-large-en-v1.5 is 1024 dims

emb_path = "../artifacts/embeddings.memmap"
ids_path = "../artifacts/ids.memmap"

# -------- DETECT ROW COUNT FROM FILE SIZE --------

file_size = os.path.getsize(emb_path)

# float32 = 4 bytes
TOTAL_ROWS = file_size // (4 * EMB_DIM)

print(f"Detected rows: {TOTAL_ROWS}")

# -------- LOAD MEMMAP FILES --------

emb = np.memmap(
    emb_path,
    dtype="float32",
    mode="r",
    shape=(TOTAL_ROWS, EMB_DIM)
)

ids = np.memmap(
    ids_path,
    dtype="int64",    # changed from int32 — matches export_embeddings.py
    mode="r",
    shape=(TOTAL_ROWS,)
)

# -------- SAVE DIRECTLY (NO FULL RAM COPY) --------

np.save("../artifacts/embeddings.npy", emb)
np.save("../artifacts/ids.npy", ids)

# -------- SAVE TITLES --------

with open("../artifacts/titles.pkl", "rb") as f:
    titles = pickle.load(f)

np.save("../artifacts/titles.npy", np.array(titles, dtype=object))

print("Conversion to NPY completed.")