import unittest

import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans

from Utils.distances import euclidean_distance
from Utils.evaluation import silhouette_score


class TestEvaluation(unittest.TestCase):

    def test_silhouette(self):
        """
        This function tests the result of the Utils.evaluation.silhouette_score function, by comparing its results against
        the results of the analogue function of the sklearn.metrics module.
        :return: The test is passed iff the results of the implemented silhouette_score function are the same as the sklearn.metrics module implementation
        """
        # Creating the Test Dataset
        print("SILHOUETTE INDEX TEST:\nImporting data...")
        X = pd.DataFrame([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

        # Applying the KMeans Algorithm
        print("Applying KMeans...")
        kmeans_results = KMeans(n_clusters=2, random_state=0).fit_predict(X)

        # Applying the DBSCAN Algorithm
        print("Applying DBSCAN...")
        dbscan_results = DBSCAN(eps=3, min_samples=2).fit_predict(X)

        # Calculating the silhouette_score with the Scikit Learn function
        print("[Scikit-Learn]\t\t   Calculating test silhouette_score...")
        sk_dbscan_results = metrics.silhouette_score(X, dbscan_results)
        sk_kmeans_results = metrics.silhouette_score(X, kmeans_results)

        # Calculating the silhouette_score with the our function
        print("[ClusteringALgorithms] Calculating test silhouette_score...")
        X_kmeans_test = X.copy()
        X_kmeans_test.loc[:, 'cluster'] = kmeans_results
        X_dbscan_test = X.copy()
        X_dbscan_test.loc[:, 'cluster'] = dbscan_results

        result_db_scan = silhouette_score(X_dbscan_test, euclidean_distance)
        result_kmeans = silhouette_score(X_kmeans_test, euclidean_distance)

        self.assertEqual(sk_dbscan_results, result_db_scan)
        self.assertEqual(sk_kmeans_results, result_kmeans)


if __name__ == '__main__':
    unittest.main()
