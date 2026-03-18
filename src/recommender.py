import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.pipeline import MoviePipeline


class MovieRecommender:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        original_data_path: str = "data/movies.csv",
        processed_data_path: str = "data/processed.csv",
        index_path: str = "data/movies_v1.index",
    ):
        self.pipeline = MoviePipeline(model_name=model_name)
        if not os.path.exists(processed_data_path):
            self.data = self.pipeline.run(original_data_path)
        else:
            self.data = pd.read_csv(processed_data_path, low_memory=False)
        if not os.path.exists(index_path):
            self.pipeline.build_index(processed_data_path, index_path)

        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)

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
        recommended_movies["score"] = scores[0]

        return recommended_movies[["id", "title", "score"]]
