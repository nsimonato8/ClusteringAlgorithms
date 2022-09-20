# !pip install modin[all]
# !pip install scikit-learn-extra
# !pip install ray

# Importing stuff
import sys
# import modin.config as cfg
# from distributed import Client
from datetime import datetime

import modin.pandas as pd
import ray
# import pandas as pd
from sklearn.metrics import silhouette_score

from Clustering.KMeansFamily.kmeansfamily import kmeans
from DataPreProcessing.importance import reduce_dimensionality
from OutliersDetection.CBOD import CBOD
from OutliersDetection.LDOF import top_n_LDOF
from Tuning.MATR import MATR
from Utils.Visualization.visualization import visualize_cluster
from Utils.distances import euclidean_distance

ray.shutdown()
ray.init()

# Importing Data
test_data = pd.read_csv("Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                        engine='python')

n_dims = pd.Series([i for i in range(8, 15)], index=[str(i) for i in range(8, 15)])

pca_data = n_dims.apply(lambda n_dim: (reduce_dimensionality(data=test_data, n_final_features=n_dim), n_dim))

# Settings
EXP_NUM = 3

settings_KMEANS = {'n_init': 10,
                   'max_iter': 500,
                   'verbose': 0,
                   'algorithm': 'lloyd',
                   'distance': euclidean_distance}

settings_LDOF = {
    'n': 10,
    'k': 10
}

settings_CBOD = {
    'epsilon': 0.005
}

param = [{'n_clusters': i} for i in range(11, 17)]

# %%time

timestamp1 = datetime.now()

aux1 = pca_data.apply(lambda data: MATR(A=kmeans, D=data[0], hyperpar=param, settings=settings_KMEANS, verbose=1,
                                        path="Data/Results/Experiments/",
                                        name=f"[{EXP_NUM}]Experiment - PCA_{data[1]}_dim-KMeans"))
aux1.name = 'MATR'
aux2 = pca_data.apply(lambda data: data[1])
aux2.name = 'PCA_dim'
aux3 = aux1.apply(lambda data: silhouette_score(X=data[list(set(data.columns) - {'cluster'})], labels=data['cluster']))
aux3.name = 'silhouette'

# result = pd.concat([aux1, aux2, aux3], axis=1, names=["MATR","PCA_dim","silhouette"])
# print(f"aux1:\n{aux1.shape}")
# aux1.info()
# print(f"\naux2:\n{aux2.shape}")
# aux2.info()
# print(f"\naux3:\n{aux3.shape}")
# aux3.info()

result = pd.concat([aux2, aux3], axis=1, names=["PCA_dim", "silhouette"])

timestamp1 = datetime.now() - timestamp1

# %%time
# Printing results
timestamp2 = datetime.now()
result["silhouette"].plot(kind="bar", xlabel="Number of dimensions after PCA", ylabel="Silhouette Score",
                          figsize=(35, 30)).get_figure().savefig(
    f'Data/Results/Experiments/PCA-KMeans_sil_score{EXP_NUM}.png')
timestamp2 = datetime.now() - timestamp2

# % % time
# Printing the clusters
timestamp3 = datetime.now()

aux1.apply(lambda res: visualize_cluster(data=res,
                                         i=EXP_NUM,
                                         cluster_or_outliers='cluster',
                                         additional=f"PCA_{len(res.columns) - 1}_dim-KMEANS_{res['cluster'].max() + 1}",
                                         path="Data/Results/Experiments/"))

timestamp3 = datetime.now() - timestamp3

# % % time
# Outlier detection JUST 8 DIMENSIONS
timestamp4 = datetime.now()

res = aux1['8']

det_ldof = top_n_LDOF(data=res, distance=euclidean_distance, n=settings_LDOF['n'], k=settings_LDOF['k'])

det_ldof.info()

timestamp4 = datetime.now() - timestamp4

# %%time
timestamp5 = datetime.now()
det_cbod = CBOD(data=res, k=res['cluster'].max() + 1, epsilon=settings_CBOD['epsilon'])

det_cbod.info()
timestamp5 = datetime.now() - timestamp5

# %%time
visualize_cluster(data=det_ldof[list(set(det_ldof.index) - {'cluster'} - {'LDOF'})],
                  i=EXP_NUM,
                  cluster_or_outliers='outlier',
                  additional=f"[LDOF]PCA_{len(det_ldof.columns) - 2}_dim-KMEANS_{det_ldof['cluster'].max() + 1}",
                  path="Data/Results/Experiments/")

# %%time
visualize_cluster(data=det_cbod[list(set(det_cbod.index) - {'cluster'})],
                  i=EXP_NUM,
                  cluster_or_outliers='outlier',
                  additional=f"[CBOD]PCA_{len(det_cbod.columns) - 1}_dim-KMEANS_{det_cbod['cluster'].max() + 1}",
                  path="Data/Results/Experiments/")

# %%time
# Printing log file
# Saving the reference of the standard output
original_stdout = sys.stdout
with open(f'Data/Results/Experiments/[Experiment PCA-KMeans-MATR]_main_log_{EXP_NUM}.txt', 'w') as f:
    sys.stdout = f
    # Reset the standard output
    print(f"PCA number of dimensions parameter:\t{n_dims}")
    print(f"KMeans number of cluster candidates:\t{range(10, 16)}")
    print(f"KMeans settings:\t{settings_KMEANS}")
    print(f"LDOF settings:\t{settings_LDOF}")
    print(f"CBOD settings:\t{settings_CBOD}")
    print(f"Time elapsed for MATR computation (all of the datasets):\t{timestamp1}")
    print(f"Time elapsed for Outlier Detection (LDOF):\t{timestamp4}")
    print(f"Time elapsed for Outlier Detection (CBOD):\t{timestamp5}")
    print(f"Time elapsed for Silhouette Scores plotting:\t{timestamp2}")
    print(f"Time elapsed for Clusters plotting:\t{timestamp3}")
sys.stdout = original_stdout
