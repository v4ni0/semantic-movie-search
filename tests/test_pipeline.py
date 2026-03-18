import unittest
import os

from src.pipeline import MoviePipeline


class TestMoviePipelineRun(unittest.TestCase):

    def setUp(self):
        self.pipeline = MoviePipeline()
        self.test_path = 'data/test_processed.csv'
    
    def tearDown(self):
        if os.path.exists(self.test_path):
            os.remove(self.test_path)

    def test_run_creates_processed_csv(self):
        raw_path = 'data/movies.csv'
        df = self.pipeline.run(raw_path, self.test_path)
        self.assertTrue(os.path.exists(self.test_path))
        self.assertEqual(df.shape[0], 24267)  
        self.assertEqual(df.shape[1], 14)

class TestMoviePipelineBuildIndex(unittest.TestCase):
    def setUp(self):
        self.pipeline = MoviePipeline()
        self.test_index_path = 'data/test_movies_v1.index'

    def tearDown(self):
        if os.path.exists(self.test_index_path):
            os.remove(self.test_index_path)

    def test_build_index_creates_faiss_index(self):
        csv_path = 'data/processed.csv'
        self.pipeline.build_index(csv_path, self.test_index_path)
        self.assertTrue(os.path.exists(self.test_index_path))