import sys
import unittest
# import modin.pandas as pd
# import modin.config as cfg
# from distributed import Client
from datetime import datetime

import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score

from Clustering.KMeansFamily.kmeansfamily import kmeans
from DataPreProcessing.importance import reduce_dimensionality
from OutliersDetection.CBOD import CBOD
from OutliersDetection.LDOF import top_n_LDOF
from Tuning.MATR import MATR
from Utils.Visualization.visualization import visualize_cluster
from Utils.algebric_op import similarity_matrix
from Utils.distances import euclidean_distance

EXP_NUM = 2


class TestOutliersDetection(unittest.TestCase):

    def test_full_KMeans(self):
        """
        PCA -> Tuning -> Clustering -> Outlier Detection

        :return: The test is passed if no exceptions are thrown during the test.
        """
        # client = Client()
        # cfg.Engine.put("Dask")

        test_data = pd.read_csv("../Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                                engine='python')
        pca_data = []
        n_dims = range(8, 15)

        for n_dim in n_dims:
            pca_data.append(reduce_dimensionality(data=test_data, n_final_features=n_dim))

        settings_KMEANS = {'n_init': 10,
                           'max_iter': 500,
                           'verbose': 0,
                           'algorithm': 'lloyd',
                           'distance': euclidean_distance}

        settings_LDOF = {
            'n': 20,
            'k': 20
        }

        settings_CBOD = {
            'epsilon': 0.005
        }

        param = [{'n_clusters': i} for i in range(11, 17)]

        result = []
        timestamp1 = datetime.now()
        i = 0
        for data in pca_data:
            result.append((MATR(A=kmeans, D=data, hyperpar=param, settings=settings_KMEANS, verbose=1,
                                name=f"[{EXP_NUM}]Experiment -PCA_{n_dims[i]}_dim-KMeans"),
                           silhouette_score(X=data[list(set(data.columns) - {'cluster'})], labels=data['cluster']),
                           n_dims[i]))
            i += 1
        timestamp1 = datetime.now() - timestamp1

        # Printing results
        timestamp2 = datetime.now()
        indexes = map(lambda x: f"PCA_{x[2]} - KMEANS_{x[0]['cluster'].max() + 1}", result)
        sil_scores = pd.Series(map(lambda x: x[1], result), index=indexes)
        sil_scores.plot(kind="bar", ylabel="Silhouette Score", figsize=(35, 30)).get_figure().savefig(
            f'../Data/Results/Experiments/PCA-KMeans_sil_score{EXP_NUM}.png')
        timestamp2 = datetime.now() - timestamp2

        i = 0
        timestamp3 = datetime.now()
        for res in result:  # Printing the clusters
            visualize_cluster(data=res[0][list(set(res[0].columns) - {'cluster'})], labels=res[0]['cluster'], i=EXP_NUM,
                              cluster_or_outliers='cluster',
                              additional=f"PCA_{range(10, 25)[i]}_dim-KMEANS_{res[0]['cluster'].max() + 1}",
                              path="../Data/Results/Experiments/")
            i += 1
        timestamp3 = datetime.now() - timestamp3

        timestamp4 = datetime.now()  # Outlier detection
        det_ldof, det_cbod = [], []
        for res in result:
            det_ldof.append(
                top_n_LDOF(data=res[0], distance=euclidean_distance, n=settings_LDOF['n'], k=settings_LDOF['k']))
            det_cbod.append(CBOD(data=res[0], k=res[0]['cluster'].max() + 1, epsilon=settings_CBOD['epsilon']))
        timestamp4 = datetime.now() - timestamp4

        # Saving the reference of the standard output
        original_stdout = sys.stdout
        with open(f'../Data/Results/Experiments/[Experiment PCA-KMeans-MATR]log_{EXP_NUM}.txt', 'w') as f:
            sys.stdout = f
            # Reset the standard output
            print(f"PCA number of dimensions parameter:\t{n_dims}")
            print(f"KMeans number of cluster candidates:\t{range(10, 16)}")
            print(f"KMeans settings:\t{settings_KMEANS}")
            print(f"LDOF settings:\t{settings_LDOF}")
            print(f"CBOD settings:\t{settings_CBOD}")
            print(f"Time elapsed for MATR computation (all of the datasets):\t{timestamp1}")
            print(f"Time elapsed for Outlier Detection (CBOD and LDOF):\t{timestamp4}")
            print(f"Time elapsed for Silhouette Scores plotting:\t{timestamp2}")
            print(f"Time elapsed for Clusters plotting:\t{timestamp3}")
            print(f"\nResults:\n{'-' * 15}")
            i = 0
            for res in result:
                index = result.index(res)
                print(f"\n{'=' * 5}TEST No. {i}{'=' * 5}")
                print(f"\tDataset dimensions:\t{res[2]}")
                print(f"\tNumber of clusters:\t{res[0]['cluster'].max() + 1}")
                print(f"\tSilhouette Score:\t{res[1]}")
                print(f"\tOutliers detected:\n\t{'^' * 3}\n")
                print(f"\t[LDOF]")
                with pd.option_context('expand_frame_repr', False):
                    print(det_ldof[index].head(n=det_ldof[index].shape[0]))
                print(f"\t[CBOD]")
                print(f"Number of outliers:\t{det_cbod[index].shape[0]}\tNumber of instances:\t{res[0].shape[0]}")
                # det_cbod[index].head(n=det_cbod[index].shape[0])
                print(f"\n\t{'^' * 3}")
                print(f"{'=' * 15}\n")
                i += 1
            print(f"\n{'-' * 15}\n")

            sys.stdout = original_stdout
        pass

    def test_distances(self):
        test_data = pd.read_csv("../Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                                engine='python')

        mat = pd.DataFrame(similarity_matrix(test_data, euclidean).values)
        mat = pd.melt(mat.assign(index=mat.index), id_vars=['index'])['value']

        mat.plot(kind="box", ylabel="Distances",
                 figsize=(20, 15), ylim=(0, 100000)).get_figure().savefig(
            f'../Distribution_euclidean_similarities.png')
