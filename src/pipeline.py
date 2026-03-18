import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class MoviePipeline:

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def run(self, raw_path:str, out_path:str = 'data/processed.csv'):
        df = pd.read_csv(raw_path, low_memory=False)
        df.drop(columns=[
            'recommendations', 'backdrop_path', 'tagline',
            'production_companies'
        ],
                inplace=True)
        df = df[df['status'] == 'Released']
        df = df[df['vote_average'] >= 5.0]
        df = df[df['vote_count'] >= 50]
        df.drop(columns=['status'], inplace=True)
        df['keywords'] = df['keywords'].fillna('').astype(str)
        df.drop(columns=['budget', 'revenue'], inplace=True)
        df['title'] = df['title'].apply(lambda x: x.strip()
                                        if isinstance(x, str) else x)
        df = df.drop_duplicates(subset=['id'], keep='first')
        df = df.dropna()
        df["title"] = df["title"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["id"])

        df['content'] = (df['overview'] + " " +
                         df['genres'].astype(str).str.replace("-", " ") + " " +
                         df['keywords']).str.strip()

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        return df

    def build_index(self, csv_path:str = 'data/processed.csv', index_path:str = 'data/movies_v1.index'):
        if not os.path.exists(csv_path):
            self.run('data/movies.csv', csv_path)
        df = pd.read_csv(csv_path)
        embeddings = self.model.encode(df['content'].tolist(),
                                       show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
