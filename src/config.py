from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "movies.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_movies.csv"

MODEL_DIR = BASE_DIR / "models"
INDEX_PATH = MODEL_DIR / "movies_v1.index"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
TOP_K = 5

MODEL_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)