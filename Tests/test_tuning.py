import unittest
from datetime import datetime

import pandas as pd
from sklearn.metrics import silhouette_score

from Clustering.DensityBased.DB import dbscan
from Clustering.KMeansFamily.kmeansfamily import kmeans
from Tuning.MATR import MATR
from Tuning.silhouette import silhouette_tuning
from Utils.algebric_op import similarity_matrix
from Utils.distances import euclidean_distance


class TestTuning(unittest.TestCase):

    def test_matr_kmeans(self):
        print("MATR TEST - [KMEANS]:\nImporting data...")
        test_data = pd.read_csv("../Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                                engine='python')  # Importing the sample data

        settings = {'n_init': 15,
                    'max_iter': 1000,
                    'verbose': 0,
                    'algorithm': 'lloyd',
                    'distance': euclidean_distance}
        param = [{'n_clusters': i} for i in range(2, 30)]

        print("Settings:")
        for s in settings:
            print(f"\t{s}: {settings[s]}")
        print("Hyper Parameters:")
        for i in param:
            for p in i:
                print(f"\t{p}: {i[p]}")

        print("Clustering...")
        timestamp = datetime.now()
        result = MATR(D=test_data, A=kmeans, settings=settings, hyperpar=param, verbose=1, name="MATR-KMeans")

        print(f"\tTime elapsed:\t{datetime.now() - timestamp}")

        score = silhouette_score(X=test_data, labels=result['cluster'])
        print(f"Score:\t{score}\n\n")
        self.assertGreaterEqual(score, 0.0)
        pass

    def test_matr_DBSCAN(self):
        print("MATR TEST - [KMEANS]:\nImporting data...")
        test_data = pd.read_csv("../Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                                engine='python')  # Importing the sample data

        settings = {'n_jobs': 10,
                    'algorithm': 'auto'}
        param = []

        for i in range(1, 15):
            for j in range(1, 15):
                param.append({'epsilon': j,
                              'minpts': i,
                              'metric': 'minkowski',
                              'p': 1,
                              'distance': euclidean_distance})

        print("Settings:")
        for s in settings:
            print(f"\t{s}: {settings[s]}")
        print("Hyper Parameters:")
        for i in param:
            for p in i:
                print(f"\t{p}: {i[p]}")

        print("Clustering...")
        sim = similarity_matrix(test_data, euclidean_distance)
        timestamp = datetime.now()
        result = MATR(D=test_data, A=dbscan, settings=settings, hyperpar=param, verbose=1, name="MATR-DBSCAN")

        print(f"\tTime elapsed:\t{datetime.now() - timestamp}")

        try:
            score = silhouette_score(X=test_data, labels=result['cluster'], metric='minkowski', p=1)
        except ValueError:
            score = -1.

        print(f"Score:\t{score}\n\n")
        self.assertGreaterEqual(score, 0.0)
        pass

    def test_elbow_kmeans(self):
        print("Silhouette TEST - [KMEANS]:\nImporting data...")
        test_data = pd.read_csv("../Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                                engine='python')  # Importing the sample data

        settings = {'n_init': 10,
                    'max_iter': 500,
                    'verbose': 0,
                    'algorithm': 'lloyd'}
        param = [{'n_clusters': i} for i in range(2, 30)]

        print("Settings:")
        for s in settings:
            print(f"\t{s}: {settings[s]}")
        print("Hyper Parameters:")
        for i in param:
            for p in i:
                print(f"\t{p}: {i[p]}")

        print("Clustering...")
        timestamp = datetime.now()
        result = silhouette_tuning(D=test_data, A=kmeans, settings=settings, hyperpar=param, verbose=1)

        print(f"\tTime elapsed:\t{datetime.now() - timestamp}")

        score = silhouette_score(X=test_data, labels=result['cluster'])
        print(f"Score:\t{score}\n\n")

        self.assertGreaterEqual(score, 0.0)
        pass

    def test_matrcv(self):
        pass


if __name__ == '__main__':
    unittest.main()
