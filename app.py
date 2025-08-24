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
FORCE_REBUILD = os.environ.get("FORCE_REBUILD", "0") == "1"


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

    if not FORCE_REBUILD and idx_path.exists() and meta_path.exists():
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

# Sanity: index vectors vs meta rows
try:
    ntotal = int(getattr(index, "ntotal", 0))
except Exception:
    ntotal = 0

if ntotal != len(meta):
    print(f"[KB][WARN] Index/meta size mismatch: index.ntotal={ntotal}, meta_rows={len(meta)}", flush=True)
else:
    print(f"[KB] Index/meta sizes OK: {ntotal}", flush=True)

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
    try:
        ntotal = int(getattr(index, "ntotal", 0))
    except Exception:
        ntotal = 0
    return {
        "status": "ok",
        "index_dir": str(INDEX_DIR),
        "model": MODEL if 'MODEL' in globals() else LOCAL_EMBED_MODEL,
        "chunks": len(meta),
        "ntotal": ntotal
    }


def _to_int_safe(v) -> int:
    try:
        # handle "12.0" or other weird strings
        s = str(v).strip()
        if "." in s:
            s = s.split(".", 1)[0]
        return int(s)
    except Exception:
        return 0

def _clean_value(v):
    if v is None: return ""
    if isinstance(v, float) and not isfinite(v): return ""
    if isinstance(v, (dict, list)): return v
    # keep ints/floats/bools as-is; everything else to string
    return v if isinstance(v, (str, int, float, bool)) else str(v)

def sanitize_meta_row(m: dict) -> dict:
    # NEVER raise — always return a JSON-safe dict
    return {
        "chunk_id": _to_int_safe(m.get("chunk_id")),
        "doc_id": _clean_value(m.get("doc_id")),
        "title": _clean_value(m.get("title")),
        "text": _clean_value(m.get("text")),
        "source_url": _clean_value(m.get("source_url")),
        "updated": _clean_value(m.get("updated")),
        "tags": _clean_value(m.get("tags")),
    }

@app.post("/search")
def search(req: SearchRequest, x_api_key: Optional[str] = Header(None)):
    # ... (auth + q + guards unchanged)

    # cap k
    try:
        k_req = int(req.k or 5)
    except Exception:
        k_req = 5
    k = max(1, min(k_req, MAX_K, len(meta)))

    # embed & search
    try:
        qv = embed_query(q)
        D, I = index.search(qv, k)
    except Exception as e:
        print(f"[KB] Search pipeline failed: {e}", flush=True)
        raise HTTPException(status_code=500, detail="search failed")

    # coerce any non-finite distances
    D = np.nan_to_num(D, nan=0.0, posinf=1.0, neginf=-1.0)

    hits = []
    neg_index = 0
    bad_meta = 0

    for score, idx in zip(D[0], I[0]):
        if idx is None or int(idx) < 0:
            neg_index += 1
            continue
        # SAFELY fetch meta row
        try:
            m_raw = meta[int(idx)]
        except Exception as e:
            bad_meta += 1
            print(f"[KB][WARN] meta lookup failed for idx={idx}: {e}", flush=True)
            continue
        m = sanitize_meta_row(m_raw)
        # ensure score is JSON-safe float
        try:
            fscore = float(score)
        except Exception:
            fscore = 0.0
        hits.append({"score": fscore, **m})

    raw_labels = len(I[0]) if I is not None and len(I) > 0 else 0
    preview = [{"score": round(h["score"], 4), "title": str(h["title"])[:80]} for h in hits[:3]]
    print(
        f"[KB] Query='{q[:60]}{'...' if len(q) > 60 else ''}' "
        f"raw={raw_labels}, ok={len(hits)}, neg_index={neg_index}, bad_meta={bad_meta}",
        flush=True
    )
    if preview:
        print(f"[KB] Top preview: {preview}", flush=True)

    return {"results": hits}


@app.get("/dev/grep")
def grep(q: str):
    ql = q.lower()
    matches = []
    for m in meta:
        if ql in str(m.get("text","")).lower():
            matches.append({"doc_id": m.get("doc_id"), "title": m.get("title")})
            if len(matches) >= 5:
                break
    return {"matches": matches, "total_scanned": len(meta)}
