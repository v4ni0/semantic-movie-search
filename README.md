# Movie Recommendation System

A content-based movie recommendation system built with Python. It uses **Sentence Transformers** to embed movie content into vectors and **FAISS** to perform fast similarity search. The project includes a data pipeline, a FastAPI backend, and a Streamlit UI.

## Features

- Content-based recommendations using semantic text embeddings (`all-MiniLM-L6-v2`)
- Fast nearest-neighbor retrieval with FAISS
- Data cleaning + feature engineering pipeline
- REST API with FastAPI (`/recommend`)
- Streamlit frontend for interactive use
- Unit tests with `pytest`

## Project Structure
```
├── data/
│   ├── processed/
│   │   └── cleaned_movies.csv
│   └── raw/
│       └── movies.csv
├── models/
│   └── movies_v1.index
├── notebooks/
│   ├── content_based_filtering.ipynb
│   └── EDA.ipynb
├── src/
│   ├── api/
│   │   └── main.py
│   ├── __init__.py
│   ├── config.py
│   ├── data_audit.py
│   ├── pipeline.py
│   └── recommender.py
├── tests/
│   ├── api/
│   │   └── test_api.py
│   ├── __init__.py
│   ├── test_pipeline.py
│   └── test_recommender.py
├── view/
│   └── app.py
├── Dockerfile
└── requirements.txt
```

## Requirements

- Python 3.9+ recommended
- Dependencies are listed in requirements.txt

## Installation

```bash
# from the repo root
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
```

## Data Setup

Place the raw dataset at:

```text
data/raw/movies.csv
```

The pipeline expects movie fields commonly found in TMDB-style datasets, including:
`id`, `title`, `overview`, `genres`, `keywords`, `status`, `vote_average`, `vote_count`.

## How It Works (High Level)

1. `MoviePipeline.run()` cleans the raw CSV and creates a `content` field from:
   - `overview`
   - `genres`
   - `keywords`
2. `MoviePipeline.build_index()` embeds `content` and builds a FAISS index (`IndexFlatIP`) using normalized vectors.
3. `MovieRecommender.recommend()` embeds the user query and retrieves the most similar movies from FAISS.

## Run the Pipeline (Optional)

The recommender will auto-generate processed data / index if missing, but you can run the pipeline manually:

```python
from src.pipeline import MoviePipeline

pipeline = MoviePipeline()

# Clean and save processed data
pipeline.run("data/raw/movies.csv")

# Build and save FAISS index
pipeline.build_index()
```

## Run the API (FastAPI)

Start the API:

```bash
python -m src.api.main
```

Or with Uvicorn:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

API docs:

- Swagger UI: `http://localhost:8000/docs`

### Endpoint

`POST /recommend`

Example request body:

```json
{
  "description": "Space travel and aliens",
  "top_k": 5
}
```

Example response shape:

```json
{
  "recommendations": [
    { "id": 123, "title": "Interstellar", "score": 0.73 }
  ]
}
```

## Run the Streamlit App

```bash
streamlit run view/app.py
```

## Testing

Run the full test suite:

```bash
pytest -q
```

## Configuration

Key configuration lives in config.py, including:

- `RAW_DATA_PATH`: movies.csv
- `PROCESSED_DATA_PATH`: cleaned_movies.csv
- `INDEX_PATH`: movies_v1.index
- `MODEL_NAME`: `all-MiniLM-L6-v2`
- `TOP_K`: default number of recommendations

## Docker (Optional)

Build:

```bash
docker build -t movie-recommender .
```

Run (API):

```bash
docker run -p 8000:8000 movie-recommender
```
