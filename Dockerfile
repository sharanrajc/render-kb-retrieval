
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY data ./data
COPY build_index_openai.py ./build_index_openai.py
COPY app.py ./app.py
COPY run.sh ./run.sh
ENV MODEL=text-embedding-3-small
ENV INDEX_DIR=/data/index_openai
EXPOSE 8000
CMD ["./run.sh"]
