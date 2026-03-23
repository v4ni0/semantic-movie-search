import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.pipeline import MoviePipeline
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, INDEX_PATH, MODEL_NAME


class MovieRecommender:

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        original_data_path: str = RAW_DATA_PATH,
        processed_data_path: str = PROCESSED_DATA_PATH,
        index_path: str = INDEX_PATH,
    ):
        self.pipeline = MoviePipeline(model_name=model_name)
        if not os.path.exists(processed_data_path):
            self.data = self.pipeline.run(original_data_path)
        else:
            self.data = pd.read_csv(processed_data_path, low_memory=False)
        if not os.path.exists(index_path):
            self.pipeline.build_index(processed_data_path, index_path)

        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(str(index_path))

    def recommend(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query = str(query).strip()
        if not query:
            raise ValueError("Please provide a valid query.")
        query_embeddings = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embeddings = np.asarray(query_embeddings, dtype="float32")
        scores, indices = self.index.search(query_embeddings, top_k)
        recommended_movies = self.data.iloc[indices[0]].copy()
        results=[]
        for i in range(len(indices[0])):
            row = recommended_movies.iloc[i]
            results.append({
                "id": int(row["id"]),
                "title": str(row["title"]),
                "score": float(scores[0][i])
            })


        return results
