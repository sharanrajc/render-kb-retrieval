
import os, json, argparse
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI

def embed_batch(client, model, texts):
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/oasis_kb_chunks.csv")
    ap.add_argument("--out_dir", default="/data/index_openai")
    ap.add_argument("--model", default=os.environ.get("EMBED_MODEL","text-embedding-3-small"))
    ap.add_argument("--batch", type=int, default=100)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.chunks)
    if df.empty or "text" not in df.columns:
        raise SystemExit("CSV empty or missing 'text' column")

    texts = df["text"].astype(str).tolist()
    client = OpenAI()

    vecs = []
    for i in range(0, len(texts), args.batch):
        batch = texts[i:i+args.batch]
        embs = embed_batch(client, args.model, batch)
        vecs.extend(embs)

    X = np.array(vecs, dtype="float32")
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, str(out_dir / "oasis_openai.index"))

    meta = df[["chunk_id","doc_id","title","source_url","updated","tags","text"]].to_dict(orient="records")
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    (out_dir / "index_config.json").write_text(json.dumps({"model": args.model, "dim": dim, "similarity": "cosine", "text_col": "text"}, indent=2))

    print("Index built at", out_dir)
if __name__ == "__main__":
    main()
