
import os, json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import numpy as np
import faiss
from openai import OpenAI

INDEX_DIR = Path(os.environ.get("INDEX_DIR", "/data/index_openai"))
MODEL = os.environ.get("MODEL", "text-embedding-3-small")
API_KEY = os.environ.get("RETRIEVER_API_KEY", "")
client = OpenAI()

def ensure_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    idx = INDEX_DIR / "oasis_openai.index"
    meta = INDEX_DIR / "meta.json"
    if idx.exists() and meta.exists():
        return
    os.environ.setdefault("EMBED_MODEL", MODEL)
    os.system("python build_index_openai.py --chunks data/oasis_kb_chunks.csv --out_dir {}".format(INDEX_DIR))

ensure_index()
index = faiss.read_index(str(INDEX_DIR / "oasis_openai.index"))
meta = json.loads((INDEX_DIR / "meta.json").read_text())

app = FastAPI(title="KB Retrieval (FAISS baked @ runtime)", version="1.0.0")

class SearchRequest(BaseModel):
    query: str
    k: int = 5

def embed_query(q: str):
    resp = client.embeddings.create(model=MODEL, input=[q])
    v = np.array(resp.data[0].embedding, dtype="float32")
    v /= (np.linalg.norm(v) + 1e-12)
    return v.reshape(1, -1)

@app.get("/health")
def health():
    return {"status":"ok", "index_dir": str(INDEX_DIR)}

@app.post("/search")
def search(req: SearchRequest, x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    qv = embed_query(req.query)
    D, I = index.search(qv, req.k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        m = meta[idx]
        hits.append({"score": float(score), **m})
    return {"results": hits}
