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
        index_path = 'data/movies_v1.index'

        with self.assertRaises(FileNotFoundError):
            MovieRecommender(data_path=invalid_csv, index_path=index_path)

    def test_when_invalid_index_path_provided_then_throws_exception(self):
        csv_path = 'data/cleaned_movies.csv'
        invalid_index = 'data/wrong_file.index'
        with self.assertRaises(Exception):
            MovieRecommender(data_path=csv_path, index_path=invalid_index)


class TestMovieRecommenderRecommend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recommender = MovieRecommender(
            model_name='all-MiniLM-L6-v2',
            original_data_path='data/movies.csv',
            processed_data_path='data/processed.csv',
            index_path='data/movies_v1.index'
        )

    def test_recommend_returns_correct_number_of_results(self):
        query = "Space travel and aliens"
        expected_count = 5
        results = self.recommender.recommend(query, top_k=expected_count)
        self.assertEqual(len(results), expected_count)

    def test_recommend_returns_expected_columns(self):
        query = "Romantic comedy"
        expected_columns = ['id', 'title', 'score']
        results = self.recommender.recommend(query)
        self.assertTrue(all(col in results.columns for col in expected_columns))

    def test_recommend_returns_interstellar_when_given_description(self):
        query = "A group of astronauts travels through a wormhole in space in an attempt to ensure humanity's survival."
        results = self.recommender.recommend(query, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results.iloc[0]['title'], 'Interstellar')
        self.assertNotEqual(results.iloc[0]['title'], 'Other Movie')