import faiss
import os

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

class MovieRecommender:
    def __init__(self, model_name='all-MiniLM-L6-v2', data_path='data/cleaned_movies.csv', index_path='notebooks/movies_v1.index'):
        if not os.path.exists(index_path) or not os.path.exists(data_path):
            raise FileNotFoundError(f"Check your paths! Index: {index_path}, Data: {data_path}")
        
        self.model = SentenceTransformer(model_name)
        self.movie_embeddings = None
        self.movies = None
        self.index = faiss.read_index(index_path)
        self.data = pd.read_csv(data_path)

    def recommend(self, query, top_k = 5):
        query_embeddings = self.model.encode([query])
        distances, indices = self.index.search(query_embeddings, top_k)
        recommended_movies = self.data.iloc[indices[0]].copy()
        recommended_movies['score'] = distances[0]
        return recommended_movies[['id', 'title', 'score']]