import os, sys, json, time, argparse, traceback
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI, OpenAIError

def log(msg):
    print(f"[BUILDER] {msg}", flush=True)

def embed_batch(client, model, texts, max_retries=5, backoff=2.0):
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except OpenAIError as e:
            log(f"OpenAI error on attempt {attempt}/{max_retries}: {e}")
        except Exception as e:
            log(f"Unknown embedding error on attempt {attempt}/{max_retries}: {e}")
        time.sleep(backoff * attempt)
    raise RuntimeError("Failed to embed after retries.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/oasis_kb_chunks.csv")
    ap.add_argument("--out_dir", default="/data/index_openai")
    ap.add_argument("--model", default=os.environ.get("EMBED_MODEL", "text-embedding-3-small"))
    ap.add_argument("--batch", type=int, default=100)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        log("OPENAI_API_KEY not set.")
        return 1

    chunks_path = Path(args.chunks)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        log(f"CSV not found at {chunks_path.resolve()}")
        return 1

    try:
        df = pd.read_csv(chunks_path)
    except Exception as e:
        log(f"CSV read failed: {e}")
        return 1

    if df.empty:
        log("CSV is empty.")
        return 1
    if "text" not in df.columns:
        log("CSV missing 'text' column.")
        return 1

    model = args.model
    log(f"Embedding model: {model}")
    log(f"Rows to embed: {len(df)}")
    log(f"Output dir: {out_dir}")

    texts = df["text"].astype(str).tolist()
    client = OpenAI()

    vecs = []
    try:
        for i in range(0, len(texts), args.batch):
            batch = texts[i:i + args.batch]
            embs = embed_batch(client, model, batch)
            vecs.extend(embs)
            if (i // args.batch) % 10 == 0:
                log(f"Progress: {min(i + args.batch, len(texts))}/{len(texts)}")
    except Exception as e:
        log(f"Embedding loop failed: {e}")
        traceback.print_exc()
        return 1

    try:
        X = np.array(vecs, dtype="float32")
        if X.ndim != 2 or X.shape[0] == 0:
            log(f"Bad embedding matrix shape: {X.shape}")
            return 1
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        dim = X.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(X)
        faiss.write_index(index, str(out_dir / "oasis_openai.index"))
        meta = df[["chunk_id","doc_id","title","source_url","updated","tags","text"]].to_dict(orient="records")
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "index_config.json").write_text(json.dumps({
            "model": model, "dim": dim, "similarity": "cosine", "text_col": "text"
        }, indent=2), encoding="utf-8")
    except Exception as e:
        log(f"FAISS/Write failed: {e}")
        traceback.print_exc()
        return 1

    log("Index built successfully.")
    return 0

if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
