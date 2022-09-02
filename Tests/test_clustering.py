import datetime
import unittest

import pandas as pd

from Clustering.DensityBased.DB import dbscan
from Utils import distances, evaluation, infogain
from Utils.distances import cosine_distance
from Utils.evaluation import silhouette_score


class TestClustering(unittest.TestCase):

    def test_kmeans(self):
        # self.assertEqual('foo'.upper(), 'FOO')
        print("KMEANS TEST:\nImporting data...")
        test_data = pd.read_csv("../TestData/session_sample.csv")  # Importing the sample data

        settings = {'iterations_max': 10}
        print(f"Settings:\n\tMaximum number of iterations: {settings['iterations_max']}")

        print("Clustering...")
        timestamp = datetime.datetime.now()
        result = Clustering.KMeansFamily.kmeansfamily.kmeans(data=test_data, similarity=distances.cosine_distance,
                                                             infogain=infogain.has_changed, verbose=1)
        print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

        print(f"Score:\t{evaluation.silhouette_score(data=result, distance=distances.cosine_distance)}")

    def test_kmedoids(self):
        print("KMEDOIDS TEST:\nImporting data...")
        test_data = pd.read_csv("../TestData/session_sample.csv")  # Importing the sample data

        settings = {'iterations_max': 10}
        print(f"Settings:\n\tMaximum number of iterations: {settings['iterations_max']}")

        print("Clustering...")
        timestamp = datetime.datetime.now()
        result = Clustering.KMeansFamily.kmeansfamily.kmedoids(data=test_data, similarity=distances.cosine_distance,
                                                               infogain=infogain.has_changed, verbose=1)
        print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

        print(f"Score:\t{evaluation.silhouette_score(data=result, distance=distances.cosine_distance)}")
        pass

    def test_dbscan(self):
        print("DBSCAN TEST:\nImporting data...")
        test_data = pd.read_csv("../TestData/session_sample.csv", header=0)  # Importing the sample data
        # test_data.drop(columns="user_id", inplace=True)

        settings = {'iterations_max': 10}
        print(f"Settings:\n\tMaximum number of iterations: {settings['iterations_max']}")

        print("Clustering...")
        result = dbscan(data=test_data, similarity=cosine_distance, verbose=1, settings=settings, epsilon=0.381,
                        minpts=2)

        try:
            print(f"Score:{silhouette_score(data=result, distance=cosine_distance)}")
        except Exception:
            print("Result is empty!")
        pass


if __name__ == '__main__':
    unittest.main()
