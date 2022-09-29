import datetime
import unittest

import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score

from Clustering.DensityBased.DB import dbscan
from Clustering.Hierarchical.HAC import HAC
from Clustering.KMeansFamily.kmeansfamily import kmeans, kmedoids
from DataPreProcessing.importance import reduce_dimensionality
from Utils.Visualization.visualization import visualize_cluster


class TestClustering(unittest.TestCase):

    def test_kmeans(self):
        # self.assertEqual('foo'.upper(), 'FOO')
        print("KMEANS TEST:\nImporting data...")
        test_data = pd.read_csv("../Data/10k_sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                                engine='python')  # Importing the sample data

        settings = {'n_init': 10,
                    'max_iter': 500,
                    'verbose': 0,
                    'algorithm': 'lloyd',
                    'distance': euclidean}

        param = {'n_clusters': 16}

        print("Settings:")
        for s in settings:
            print(f"\t{s}: {settings[s]}")
        print("Hyper Parameters:")
        for p in param:
            print(f"\t{p}: {param[p]}")

        n_dims = pd.Series([8, 10, 12, 16], index=[str(i) for i in [8, 10, 12, 16]])
        pca_data = n_dims.apply(lambda n_dim: reduce_dimensionality(data=test_data, n_final_features=n_dim))

        print("Clustering...")
        timestamp = datetime.datetime.now()
        result = pca_data.apply(lambda x: kmeans(data=x, hyperpar=param, settings=settings))
        print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

        result.apply(lambda x: visualize_cluster(data=x,
                                                 i=1,
                                                 cluster_or_outliers='cluster',
                                                 additional=f"[PCA_{len(x.columns) - 1}_dim-KMEANS_{x['cluster'].max() + 1}]10k_",
                                                 path="../Data/Results/"))

    def test_HAC(self):
        # self.assertEqual('foo'.upper(), 'FOO')
        print("HAC TEST:\nImporting data...")
        test_data = pd.read_csv("../Data/10k_sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                                engine='python')  # Importing the sample data

        settings = {'compute_full_tree': 'auto',
                    'linkage': 'ward',
                    'distance': 'euclidean',
                    'epsilon': None}
        param = {'n_clusters': 11}

        print("Settings:")
        for s in settings:
            print(f"\t{s}: {settings[s]}")
        print("Hyper Parameters:")
        for p in param:
            print(f"\t{p}: {param[p]}")

        n_dims = pd.Series([8, 10, 12, 16], index=[str(i) for i in [8, 10, 12, 16]])
        pca_data = n_dims.apply(lambda n_dim: reduce_dimensionality(data=test_data, n_final_features=n_dim))

        print("Clustering...")
        timestamp = datetime.datetime.now()
        result = pca_data.apply(lambda x: HAC(data=x, hyperpar=param, settings=settings))
        print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

        result.apply(lambda x: visualize_cluster(data=x,
                                                 i=1,
                                                 cluster_or_outliers='cluster',
                                                 additional=f"[PCA_{len(x.columns) - 1}_dim-HAC_{x['cluster'].max() + 1}]10k_",
                                                 path="../Data/Results/"))

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

        settings = {'n_jobs': -1,
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
