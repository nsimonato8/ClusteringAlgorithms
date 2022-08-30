import datetime
import unittest

import pandas as pd

import KMeansFamily.kmeansfamily
from DensityBased.DB import dbscan
from Utils import distances, evaluation, infogain
from Utils.distances import cosine_distance
from Utils.evaluation import silhouette_score


class TestClustering(unittest.TestCase):

    def test_kmeans(self):
        # self.assertEqual('foo'.upper(), 'FOO')
        print("Importing data...")
        timestamp = datetime.datetime.now()
        test_data = pd.read_csv("../TestData/session_sample.csv")  # Importing the sample data
        print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

        settings = {'iterations_max': 10}
        print(f"Settings:\n\tMaximum number of iterations: {settings['iterations_max']}")

        print("Clustering...")
        timestamp = datetime.datetime.now()
        result = KMeansFamily.kmeansfamily.kmedoids(data=test_data, similarity=distances.cosine_distance,
                                                    infogain=infogain.has_changed, verbose=1)
        print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

        timestamp = datetime.datetime.now()
        print(
            f"Result:\n{result.info()}\nScore:{evaluation.silhouette_score(data=result, distance=distances.cosine_distance)}\nTime elapsed: {datetime.datetime.now() - timestamp}")
        pass

    def test_kmedoids(self):
        print("Importing data...")
        timestamp = datetime.datetime.now()
        test_data = pd.read_csv("../TestData/session_sample.csv")  # Importing the sample data
        print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

        settings = {'iterations_max': 10}
        print(f"Settings:\n\tMaximum number of iterations: {settings['iterations_max']}")

        print("Clustering...")
        timestamp = datetime.datetime.now()
        result = KMeansFamily.kmeansfamily.kmedoids(data=test_data, similarity=distances.cosine_distance,
                                                    infogain=infogain.has_changed, verbose=1)
        print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

        timestamp = datetime.datetime.now()
        print(
            f"Result:\n{result.info()}\nScore:{evaluation.silhouette_score(data=result, distance=distances.cosine_distance)}\nTime elapsed: {datetime.datetime.now() - timestamp}")
        pass

    def test_dbscan(self):
        print("Importing data...")
        timestamp = datetime.datetime.now()
        test_data = pd.read_csv("../TestData/session_sample.csv", header=0)  # Importing the sample data
        test_data.drop(columns="user_id", inplace=True)
        print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

        settings = {'iterations_max': 10}
        print(f"Settings:\n\tMaximum number of iterations: {settings['iterations_max']}")

        print("Clustering...")
        timestamp = datetime.datetime.now()
        result = dbscan(data=test_data, similarity=cosine_distance, verbose=1, settings=settings, epsilon=0.381,
                        minpts=2)
        print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

        timestamp = datetime.datetime.now()
        try:
            print(
                f"Score:{silhouette_score(data=result, distance=cosine_distance)}\nTime elapsed: {datetime.datetime.now() - timestamp}")
        except Exception:
            print(f"Result is empty!\tTime elapsed: {datetime.datetime.now() - timestamp}")
        pass


if __name__ == '__main__':
    unittest.main()
