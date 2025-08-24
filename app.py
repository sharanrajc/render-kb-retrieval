import os
import json
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

import numpy as np
import faiss
from openai import OpenAI


# -----------------------
# Config / Environment
# -----------------------
INDEX_DIR = Path(os.environ.get("INDEX_DIR", "/data/index_openai"))
MODEL = os.environ.get("MODEL", "text-embedding-3-small")
API_KEY = os.environ.get("RETRIEVER_API_KEY", "")  # optional API key for /search
CSV_PATH = Path("data/oasis_kb_chunks.csv")

# OpenAI client (requires OPENAI_API_KEY in env)
client = OpenAI()


# -----------------------
# Helpers
# -----------------------
def ensure_index() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    idx_path = INDEX_DIR / "oasis_openai.index"
    meta_path = INDEX_DIR / "meta.json"

    if idx_path.exists() and meta_path.exists():
        print(f"[KB] Using cached index at: {INDEX_DIR}", flush=True)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("[KB][FATAL] OPENAI_API_KEY is not set. Set it in Render → Environment.")

    if not CSV_PATH.exists():
        raise RuntimeError(f"[KB][FATAL] CSV not found at {CSV_PATH.resolve()} — "
                           f"make sure your repo contains data/oasis_kb_chunks.csv")

    print("[KB] Index not found. Building now…", flush=True)
    cmd = ["python", "build_index_openai.py", "--chunks", str(CSV_PATH), "--out_dir", str(INDEX_DIR)]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, flush=True)

    if proc.returncode != 0:
        # Fallback: create an empty FAISS index so the service still runs
        try:
            print("[KB] Builder failed; attempting empty index fallback…", flush=True)
            from openai import OpenAI
            import numpy as np, faiss, json
            # probe dimension from a dummy embedding
            probe = OpenAI().embeddings.create(model=MODEL, input=["dimension probe"]).data[0].embedding
            dim = len(probe)
            index = faiss.IndexFlatIP(dim)
            faiss.write_index(index, str(idx_path))
            (meta_path).write_text("[]", encoding="utf-8")
            (INDEX_DIR / "index_config.json").write_text(json.dumps({
                "model": MODEL, "dim": dim, "similarity": "cosine", "text_col": "text"
            }, indent=2), encoding="utf-8")
            print("[KB] Empty index created; service will respond with zero hits until CSV is populated.", flush=True)
        except Exception as e:
            raise RuntimeError(f"[KB][FATAL] Index build failed and empty-index fallback also failed: {e}")
    else:
        print("[KB] Index build complete.", flush=True)


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query using OpenAI, return a normalized (1, dim) vector for cosine/IP search.
    """
    if not text.strip():
        raise ValueError("empty query")
    # Note: client must have OPENAI_API_KEY available in env.
    resp = client.embeddings.create(model=MODEL, input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    vec /= (np.linalg.norm(vec) + 1e-12)
    return vec.reshape(1, -1)


# -----------------------
# App startup
# -----------------------
print(f"[KB] Booting with MODEL={MODEL}, INDEX_DIR={INDEX_DIR}", flush=True)
ensure_index()  # Build if needed (first boot), or load cached (subsequent boots)
print("[KB] Loading FAISS index & metadata…", flush=True)

index = faiss.read_index(str((INDEX_DIR / "oasis_openai.index")))
meta = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))

print(f"[KB] Ready. Index size: {len(meta)} chunks.", flush=True)


# -----------------------
# FastAPI
# -----------------------
app = FastAPI(title="KB Retrieval (FAISS baked @ runtime)", version="1.0.0")


class SearchRequest(BaseModel):
    query: str
    k: int = 5


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_dir": str(INDEX_DIR),
        "model": MODEL,
        "chunks": len(meta),
    }


@app.post("/search")
def search(req: SearchRequest, x_api_key: Optional[str] = Header(None)):
    # Optional header key (enable by setting RETRIEVER_API_KEY)
    if API_KEY and x_api_key != API_KEY:
        print("[KB] Unauthorized search attempt (bad x-api-key).", flush=True)
        raise HTTPException(status_code=401, detail="unauthorized")

    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is required")

    try:
        qv = embed_query(q)
    except Exception as e:
        print(f"[KB] Embed failed: {e}", flush=True)
        raise HTTPException(status_code=500, detail="embedding failed")

    try:
        D, I = index.search(qv, req.k)
    except Exception as e:
        print(f"[KB] FAISS search failed: {e}", flush=True)
        raise HTTPException(status_code=500, detail="search failed")

    hits = []
    for score, idx in zip(D[0], I[0]):
        try:
            m = meta[idx]
            hits.append({"score": float(score), **m})
        except Exception:
            # If meta/index ever drift, skip bad rows
            continue

    print(f"[KB] Query='{q[:60]}{'...' if len(q) > 60 else ''}' → {len(hits)} hits", flush=True)
    return {"results": hits}
