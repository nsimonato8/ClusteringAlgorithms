import sys
import unittest
# import modin.pandas as pd
from datetime import datetime

import pandas as pd
from sklearn.metrics import silhouette_score

from Clustering.KMeansFamily.kmeansfamily import kmeans
from DataPreProcessing.importance import reduce_dimensionality
from Tuning.MATR import MATR
# import modin.config as cfg
# from distributed import Client
from Utils.Visualization.visualization import visualize_cluster
from Utils.distances import euclidean_distance


class TestOutliersDetection(unittest.TestCase):

    def test_full_KMeans(self):
        """

        :return: The test is passed if no exception are thrown during the test.
        """
        # client = Client()
        # cfg.Engine.put("Dask")

        test_data = pd.read_csv("../Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                                engine='python')
        pca_data = []
        n_dims = range(10, 25)

        for n_dim in range(10, 25):
            pca_data.append(reduce_dimensionality(data=test_data, n_final_features=n_dim))

        settings = {'n_init': 10,
                    'max_iter': 500,
                    'verbose': 0,
                    'algorithm': 'lloyd',
                    'distance': euclidean_distance}
        param = [{'n_clusters': i} for i in range(10, 16)]

        result = []
        timestamp1 = datetime.now()
        i = 0
        for data in pca_data:
            result.append((MATR(A=kmeans, D=data, hyperpar=param, settings=settings, verbose=1,
                                name=f"Experiment -PCA_{n_dims[i]}_dim-KMeans"),
                           silhouette_score(X=data, labels=data['cluster']), n_dims[i]))
            i += 1
        timestamp1 = datetime.now() - timestamp1

        # Printing results
        timestamp2 = datetime.now()
        indexes = map(lambda x: f"PCA_{x[2]} - KMEANS_{x[0]['cluster'].max() + 1}", result)
        sil_scores = pd.Series(map(lambda x: x[1], result), index=indexes)
        sil_scores.plot(kind="bar", ylabel="Silhouette Score", figsize=(15, 10)).get_figure().savefig(
            f'PCA-KMeans_sil_score{0}.png')
        timestamp2 = datetime.now() - timestamp2

        i = 0
        timestamp3 = datetime.now()
        for res in result:  # Printing the clusters
            visualize_cluster(data=res[0], labels=res[0]['cluster'], i=i, cluster_or_outliers='cluster',
                              additional=f"[PCA_{range(10, 25)[i]}_dim-KMEANS]", path="../Data/Results/Experiments")
            i += 1
        timestamp3 = datetime.now() - timestamp3
        # Saving the reference of the standard output
        original_stdout = sys.stdout
        with open(f'[Experiment PCA-KMeans-MATR]log_{0}.txt', 'w') as f:
            sys.stdout = f
            # Reset the standard output
            print(f"PCA number of dimensions parameter:\t{n_dims}")
            print(f"KMeans number of cluster candidates:\t{range(10, 16)}")
            print(f"KMeans settings:\t{settings}")
            print(f"Time elapsed for MATR computation (all of the datasets):\t{timestamp1}")
            print(f"Time elapsed for Silhouette Scores plotting:\t{timestamp2}")
            print(f"Time elapsed for Clusters plotting:\t{timestamp3}")
            print(f"\nResults:\n{'-' * 15}")
            for res in result:
                print(f"\tDataset dimensions:\t{res[2]}")
                print(f"\tNumber of clusters:\t{res[0]['cluster'].max() + 1}")
                print(f"\tSilhouette Score:\t{res[1]}")
            print(f"\n{'-' * 15}")
            sys.stdout = original_stdout
        pass
