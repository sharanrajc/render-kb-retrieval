#!/usr/bin/env python3
# build_index_openai.py
# Builds a FAISS index from data/oasis_kb_chunks.csv using OpenAI embeddings.
# Adds build-time hygiene: NaN cleanup and strict JSON (allow_nan=False).

import os, sys, json, time, argparse, traceback
from pathlib import Path

import pandas as pd
import numpy as np
import faiss
from openai import OpenAI, OpenAIError


def log(msg: str) -> None:
    print(f"[BUILDER] {msg}", flush=True)


def embed_batch(client: OpenAI, model: str, texts, max_retries: int = 5, backoff: float = 2.0):
    """Embed a batch with retries for transient OpenAI errors."""
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Build FAISS index for Oasis KB using OpenAI embeddings")
    ap.add_argument("--chunks", default="data/oasis_kb_chunks.csv", help="CSV path")
    ap.add_argument("--out_dir", default="/data/index_openai", help="Output directory for index+meta")
    ap.add_argument("--model", default=os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
                    help="OpenAI embedding model")
    ap.add_argument("--batch", type=int, default=100, help="Embedding batch size")
    ap.add_argument("--min-chars", type=int, default=200, help="Discard rows with text shorter than this")
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

    # ---- Load & sanitize CSV ----
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

    # Ensure required columns exist (create if missing)
    for col in ["chunk_id", "doc_id", "title", "source_url", "updated", "tags"]:
        if col not in df.columns:
            df[col] = ""

    # Convert NaN -> "" for all non-numeric cols to avoid NaNs in JSON/meta
    df = df.replace({np.nan: ""})

    # Drop very short rows (build-time hygiene)
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() >= int(args.min_chars)].reset_index(drop=True)
    if df.empty:
        log("After min-chars filtering, no rows remain.")
        return 1

    # Best-effort types
    # chunk_id numeric if possible; else assign sequential ids
    try:
        df["chunk_id"] = pd.to_numeric(df["chunk_id"], errors="coerce")
    except Exception:
        df["chunk_id"] = np.nan
    if df["chunk_id"].isna().any():
        df["chunk_id"] = range(1, len(df) + 1)

    # ---- Embeddings ----
    model = args.model
    log(f"Embedding model: {model}")
    log(f"Rows to embed: {len(df)}")
    log(f"Output dir: {out_dir}")

    client = OpenAI()

    texts = df["text"].astype(str).tolist()
    vecs = []
    try:
        for i in range(0, len(texts), args.batch):
            batch = texts[i:i + args.batch]
            embs = embed_batch(client, model, batch)
            # Convert to float32 now
            vecs.append(np.array(embs, dtype="float32"))
            if (i // args.batch) % 10 == 0:
                log(f"Progress: {min(i + args.batch, len(texts))}/{len(texts)}")
    except Exception as e:
        log(f"Embedding loop failed: {e}")
        traceback.print_exc()
        return 1

    # ---- Build FAISS index ----
    try:
        X = np.vstack(vecs) if len(vecs) > 1 else vecs[0]
        if X.ndim != 2 or X.shape[0] == 0:
            log(f"Bad embedding matrix shape: {X.shape}")
            return 1

        # Normalize for cosine via inner product
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        dim = X.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(X)
        faiss.write_index(index, str(out_dir / "oasis_openai.index"))
    except Exception as e:
        log(f"FAISS build/write failed: {e}")
        traceback.print_exc()
        return 1

    # ---- Write meta with strict JSON ----
    try:
        # Re-clean any accidental NaNs (paranoia)
        df_meta = df[["chunk_id", "doc_id", "title", "source_url", "updated", "tags", "text"]].copy()
        df_meta = df_meta.replace({np.nan: ""})
        # Cast chunk_id to int for consistency
        try:
            df_meta["chunk_id"] = df_meta["chunk_id"].astype(int)
        except Exception:
            # fallback to sequential ints
            df_meta["chunk_id"] = range(1, len(df_meta) + 1)

        meta = df_meta.to_dict(orient="records")

        # STRICT JSON: allow_nan=False ensures no NaN/Inf slip into meta.json
        (out_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8"
        )

        (out_dir / "index_config.json").write_text(json.dumps({
            "model": model,
            "dim": dim,
            "similarity": "cosine",
            "text_col": "text"
        }, indent=2), encoding="utf-8")

    except Exception as e:
        log(f"Meta write failed: {e}")
        traceback.print_exc()
        return 1

    log(f"Index built successfully at {out_dir} with dim={dim}, rows={len(df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
