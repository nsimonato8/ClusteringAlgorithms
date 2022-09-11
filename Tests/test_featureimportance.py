import unittest

import pandas as pd

from Clustering.KMeansFamily.kmeansfamily import kmeans
from DataPreProcessing.Visualization.visualization import visualize_cluster
from DataPreProcessing.importance import reduce_dimensionality


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
            visualize_cluster(X, result['cluster'], i=i - 10, additional=f"PCA-KMEANS with {i} dims",
                              path="../Data/Results/PCA/")

        pass


if __name__ == '__main__':
    unittest.main()
