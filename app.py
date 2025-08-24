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

from math import isfinite
MAX_K = 20  # hard cap to avoid latency/memory issues


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


def _clean_value(v):
    if v is None:
        return ""
    if isinstance(v, float) and not isfinite(v):  # NaN/Inf
        return ""
    if isinstance(v, (dict, list)):
        return v
    return str(v) if not isinstance(v, (str, int, float, bool)) else v

def sanitize_meta_row(m: dict) -> dict:
    return {
        "chunk_id": int((m.get("chunk_id") or 0)),
        "doc_id": _clean_value(m.get("doc_id")),
        "title": _clean_value(m.get("title")),
        "text": _clean_value(m.get("text")),
        "source_url": _clean_value(m.get("source_url")),
        "updated": _clean_value(m.get("updated")),
        "tags": _clean_value(m.get("tags")),
    }

@app.post("/search")
def search(req: SearchRequest, x_api_key: Optional[str] = Header(None)):
    # API key check (optional)
    if API_KEY and x_api_key != API_KEY:
        print("[KB] Unauthorized search attempt (bad x-api-key).", flush=True)
        raise HTTPException(status_code=401, detail="unauthorized")

    # define q here
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is required")

    # Guard: empty index
    total = len(meta)
    if total == 0:
        print("[KB] Search on empty index — returning zero hits.", flush=True)
        return {"results": []}

    # Cap k
    try:
        k_req = int(req.k or 5)
    except Exception:
        k_req = 5
    k = max(1, min(k_req, MAX_K, total))

    # Embed & search
    try:
        qv = embed_query(q)            # your existing embed_query() function
        D, I = index.search(qv, k)     # FAISS IP/cosine index
    except Exception as e:
        print(f"[KB] Search pipeline failed: {e}", flush=True)
        raise HTTPException(status_code=500, detail="search failed")

    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx is None or int(idx) < 0:
            continue
        try:
            fscore = float(score)
        except Exception:
            continue
        if not isfinite(fscore):
            continue
        try:
            m = sanitize_meta_row(meta[int(idx)])
        except Exception:
            continue
        hits.append({"score": fscore, **m})

    print(f"[KB] Query='{q[:60]}{'...' if len(q) > 60 else ''}' → {len(hits)} hits (k={k}, total={total})", flush=True)
    return {"results": hits}


