import unittest
from src.recommender import MovieRecommender


class TestMovieRecommenderInit(unittest.TestCase):
    def test_when_valid_paths_provided_then_object_is_successfully_initialized(self):        
        recommender = MovieRecommender()
        self.assertIsNotNone(recommender.model)
        self.assertIsNotNone(recommender.index)
        self.assertFalse(recommender.data.empty)

    def test_when_invalid_csv_path_provided_then_throws_file_not_found_error(self):

        invalid_csv = 'data/non_existent.csv'
        index_path = 'notebooks/movies_v1.index'

        with self.assertRaises(FileNotFoundError):
            MovieRecommender(data_path=invalid_csv, index_path=index_path)

    def test_when_invalid_index_path_provided_then_throws_exception(self):
        csv_path = 'data/cleaned_movies.csv'
        invalid_index = 'notebooks/wrong_file.index'
        with self.assertRaises(Exception):
            MovieRecommender(data_path=csv_path, index_path=invalid_index)