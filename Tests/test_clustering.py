import datetime
import unittest

import pandas as pd
from sklearn.metrics import silhouette_score

from Clustering.DensityBased.DB import dbscan
from Clustering.KMeansFamily.kmeansfamily import kmeans, kmedoids
from DataPreProcessing.Visualization.visualization import visualize_cluster


class TestClustering(unittest.TestCase):

    def test_kmeans(self):
        # self.assertEqual('foo'.upper(), 'FOO')
        print("KMEANS TEST:\nImporting data...")
        test_data = pd.read_csv("../Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                                engine='python')  # Importing the sample data

        settings = {'n_init': 10,
                    'max_iter': 500,
                    'verbose': 0,
                    'algorithm': 'lloyd'}
        param = {'n_clusters': 5}
        print("Settings:")
        for s in settings:
            print(f"\t{s}: {settings[s]}")
        print("Hyper Parameters:")
        for p in param:
            print(f"\t{p}: {param[p]}")

        print("Clustering...")
        timestamp = datetime.datetime.now()
        result = kmeans(data=test_data, hyperpar=param, settings=settings)
        print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

        visualize_cluster(data=test_data, labels=result['cluster'], i=1, h=3)

        score = silhouette_score(X=test_data, labels=result['cluster'])
        print(f"Score:\t{score}\n\n")
        self.assertGreaterEqual(score, 0.0)

    def test_kmedoids(self):
        print("KMEDOIDS TEST:\nImporting data...")
        test_data = pd.read_csv("../Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                                engine='python')  # Importing the sample data

        settings = {'max_iter': 500,
                    'method': 'alternate',
                    'init': 'heuristic',
                    'random_state': None}
        param = {'n_clusters': 8,
                 'metric': 'euclidean'}

        print("Settings:")
        for s in settings:
            print(f"\t{s}: {settings[s]}")
        print("Hyper Parameters:")
        for p in param:
            print(f"\t{p}: {param[p]}")

        print("Clustering...")
        timestamp = datetime.datetime.now()
        result = kmedoids(data=test_data, hyperpar=param, settings=settings)
        print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

        score = silhouette_score(X=test_data, labels=result['cluster'])
        print(f"Score:\t{score}\n\n")
        self.assertGreaterEqual(score, 0.0)
        pass

    def test_dbscan(self):
        print("DBSCAN TEST:\nImporting data...")
        test_data = pd.read_csv("../Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                                engine='python')  # Importing the sample data
        # test_data.drop(columns="user_id", inplace=True)

        settings = {'n_jobs': 10,
                    'algorithm': 'auto'}
        param = {'epsilon': 5,
                 'minpts': 5,
                 'metric': 'minkowski',
                 'p': 2}

        print("Settings:")
        for s in settings:
            print(f"\t{s}: {settings[s]}")
        print("Hyper Parameters:")
        for p in param:
            print(f"\t{p}: {param[p]}")

        print("Clustering...")
        timestamp = datetime.datetime.now()
        result = dbscan(data=test_data, hyperpar=param, settings=settings)

        print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

        score = silhouette_score(X=test_data, labels=result['cluster'])
        print(f"Score:\t{score}\n\n")
        self.assertGreaterEqual(score, -1.0)
        pass


if __name__ == '__main__':
    unittest.main()
