import unittest

import pandas as pd

from Clustering.KMeansFamily.kmeansfamily import kmeans
from DataPreProcessing.importance import reduce_dimensionality
from Utils.Visualization.visualization import visualize_cluster
from Utils.algebric_op import similarity_matrix, construct_clustering_matrix, inverse_matrix, trace
from Utils.distances import euclidean_distance


class TestFeatureImportance(unittest.TestCase):

    def test_PCA_kmeans(self):
        test_data = pd.read_csv("../Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                                engine='python')
        settings = {'n_init': 10,
                    'max_iter': 500,
                    'verbose': 0,
                    'algorithm': 'lloyd'}
        param = {'n_clusters': 5}

        score = []
        for i in range(10, 26):
            X = reduce_dimensionality(test_data, i)
            print(f"Reducing to {i} dimensions")
            result = kmeans(data=X, hyperpar=param, settings=settings)
            print(f"[{i} dimensions]:\n{result['cluster'].describe()}\n")
            visualize_cluster(X, result['cluster'], additional=f"PCA-KMEANS with {i} dims",
                              path="../Data/Results/PCA/")

        pass

    def test_PCA_kmeans_MATR_plot(self):
        test_data = pd.read_csv("../Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                                engine='python')
        settings = {'n_init': 10,
                    'max_iter': 500,
                    'verbose': 0,
                    'algorithm': 'lloyd',
                    'distance': euclidean_distance}
        param = {'n_clusters': 16}

        result = []
        for i in range(10, 26):
            X = reduce_dimensionality(test_data, i)
            S = similarity_matrix(X, settings['distance'])
            res = []
            Z = kmeans(data=X, hyperpar=param, settings=settings)
            res.append(Z)
            Z = construct_clustering_matrix(Z)
            X = S.T.dot(Z.dot(inverse_matrix(Z.T.dot(Z)).dot(Z.T)))
            result.append((i, trace(X)))

        indexes = map(lambda x: f"PCA_{x[0]} - KMEANS_16", result)
        trace_crits = pd.Series(map(lambda x: x[1], result), index=indexes)
        trace_crits.plot(kind="bar", ylabel="Trace Criterion", xlabel="Number of dimensions",
                         figsize=(35, 30)).get_figure().savefig(
            f'../Data/Results/Experiments/PCA-KMeans16_trace_crit_PLOT.png')
        pass


if __name__ == '__main__':
    unittest.main()
