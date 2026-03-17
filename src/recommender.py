import faiss

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

class MovieRecommender:
    def __init__(self, model_name='all-MiniLM-L6-v2', data_path='data/cleaned_movies.csv'):
        self.model = SentenceTransformer(model_name)
        self.movie_embeddings = None
        self.movies = None
