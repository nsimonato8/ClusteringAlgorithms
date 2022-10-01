import os
import sys
import warnings
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import ray
# import pandas as pd
from scipy.spatial.distance import euclidean

from Clustering.KMeansFamily.kmeansfamily import kmeans
from DataPreProcessing.importance import reduce_dimensionality
from OutliersDetection.CBOD import CBOD
from Tuning.MATR import MATR
from Utils.Visualization.visualization import visualize_cluster

os.environ["MODIN_CPUS"] = "20"
os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray
ray.shutdown()
ray.init(num_cpus=20)

import modin.pandas as pd

master_timestamp = datetime.now()

FILENAME = "10k_"

# Importing Data
print(f"[{datetime.now()}]IMPORTING DATA...")
test_data = pd.read_csv(f"Data/{FILENAME}sessions_cleaned.csv", sep=",", skipinitialspace=True,
                        skipfooter=3)  # , engine='python')

print(f"[{datetime.now()}]REDUCING DIMENSIONALITY...")
n_dims = pd.Series([i for i in range(8, 15)], index=[str(i) for i in range(8, 15)])

pca_data = n_dims.apply(lambda n_dim: (reduce_dimensionality(data=test_data, n_final_features=n_dim), n_dim))

# Settings
print(f"[{datetime.now()}]GENERATING SETTINGS...")
EXP_NUM = 5

settings_KMEANS = {'n_init': 10,
                   'max_iter': 500,
                   'verbose': 0,
                   'algorithm': 'lloyd',
                   'distance': euclidean}

settings_CBOD = {
    'epsilon': 0.000005
}

print(f"[{datetime.now()}]GENERATING HYPERPARAMETERS CANDIDATES...")
param = [{'n_clusters': i} for i in range(11, 17)]

print(f"[{datetime.now()}]STARTING MATR...")
timestamp1 = datetime.now()
aux1 = pca_data.apply(lambda data: MATR(A=kmeans, D=data[0], hyperpar=param, settings=settings_KMEANS, verbose=1,
                                        path="Data/Results/Experiments/KMEANS/",
                                        name=f"[{EXP_NUM}]Experiment - PCA_{data[1]}_dim-KMeans{FILENAME}"))
aux1.name = 'MATR'

timestamp1 = datetime.now() - timestamp1
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp1}...")

# print(f"[{datetime.now()}]CALCULATING SILHOUETTE SCORES...")
# timestamp3 = datetime.now()
#
# aux3 = aux1.apply(lambda data: silhouette_score(X=data[list(set(data.columns) - {'cluster'})], labels=data['cluster']))
# aux3.name = 'silhouette'
#
# timestamp3 = datetime.now() - timestamp3
# print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp3}...")
#
# print(f"[{datetime.now()}]PRINTING SILHOUETTE SCORES TO FILE...")
# timestamp2 = datetime.now()
# aux3.plot(kind="bar", xlabel="Number of dimensions after PCA", ylabel="Silhouette Score",
#           figsize=(35, 30)).get_figure().savefig(
#     f'Data/Results/Experiments/KMEANS/PCA-KMeans_sil_score{EXP_NUM}{FILENAME}.png')
# timestamp2 = datetime.now() - timestamp2
# print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp2}...")

res = aux1['8']

print(f"[{datetime.now()}]{'=' * 5} CBOD {'=' * 5}")
timestamp5 = datetime.now()
det_cbod = CBOD(data=res, k=res['cluster'].max() + 1, epsilon=settings_CBOD['epsilon'])

timestamp5 = datetime.now() - timestamp5
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp5}...")
print(f"[{datetime.now()}]{'=' * 5}---{'=' * 5}")

# Printing log file
# Saving the reference of the standard output
original_stdout = sys.stdout
with open(f'Data/Results/Experiments/KMEANS/[Experiment PCA-KMeans-MATR]_main_log_{EXP_NUM}{FILENAME}.txt', 'w') as f:
    sys.stdout = f
    # Reset the standard output
    print(f"PCA number of dimensions parameter:\n{n_dims}\n")
    print(f"KMeans number of cluster candidates:\t{range(10, 16)}")
    print(f"KMeans settings:\t{settings_KMEANS}")
    print(f"CBOD settings:\t{settings_CBOD}")
    print(f"Time elapsed for MATR computation (all of the datasets):\t{timestamp1}")
    print(f"Time elapsed for Outlier Detection (CBOD):\t{timestamp5}")
    # print(f"Time elapsed for Silhouette Scores computations:\t{timestamp3}")
    # print(f"Time elapsed for Silhouette Scores plotting:\t{timestamp2}")
sys.stdout = original_stdout

print(f"[{datetime.now()}]PRINTING CLUSTERING PAIRPLOTS TO FILES...")
# Printing the clusters
timestamp6 = datetime.now()

visualize_cluster(data=res,
                  i=EXP_NUM,
                  cluster_or_outliers='cluster',
                  additional=f"PCA_{len(res.columns) - 1}_dim-KMEANS_{res['cluster'].max() + 1}{FILENAME}",
                  path="Data/Results/Experiments/KMEANS/")

visualize_cluster(data=det_cbod[list(set(det_cbod.columns) - {'cluster'})],
                  i=EXP_NUM,
                  cluster_or_outliers='outlier',
                  additional=f"[CBOD]PCA_{len(det_cbod.columns) - 1}_dim-KMEANS_{det_cbod['cluster'].max() + 1}{FILENAME}_{EXP_NUM}",
                  path="Data/Results/Experiments/KMEANS/")
timestamp6 = datetime.now() - timestamp6
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp6}...")

print(f"[{datetime.now()}]PRINTING RESULTS TO FILES...")
timestamp7 = datetime.now()

path_results = "Data/Results/Experiments/KMEANS/"

det_cbod.to_csv(path_results + f"KMEANS_Outliers_CBOD{FILENAME}{EXP_NUM}.csv")

timestamp7 = datetime.now() - timestamp7
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp7}...")

master_timestamp = datetime.now() - master_timestamp
print(f"[{datetime.now()}]EXPERIMENT {EXP_NUM} CONCLUDED! Time elapsed:\t{master_timestamp}...")
