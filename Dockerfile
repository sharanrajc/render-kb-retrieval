FROM python:3.11-slim

# --- OS deps for faiss-cpu ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App files
COPY data ./data
COPY build_index_openai.py ./build_index_openai.py
COPY app.py ./app.py
COPY run.sh ./run.sh

# Make sure the script is executable
RUN chmod +x /app/run.sh

# Runtime env (Render will set PORT, we respect it in run.sh)
ENV MODEL=text-embedding-3-small
ENV INDEX_DIR=/data/index_openai

EXPOSE 8000
CMD ["bash","/app/run.sh"]

