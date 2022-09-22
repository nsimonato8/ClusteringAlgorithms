import os
import sys
import warnings
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import ray
# import pandas as pd

from DataPreProcessing.importance import reduce_dimensionality
from Utils.Visualization.visualization import visualize_cluster

os.environ["MODIN_CPUS"] = "20"
os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray
ray.shutdown()
ray.init(num_cpus=20)

import modin.pandas as pd

# Importing Data
print(f"[{datetime.now()}]IMPORTING DATA...")
test_data = pd.read_csv("Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                        engine='python')

print(f"[{datetime.now()}]REDUCING DIMENSIONALITY...")
n_dims = pd.Series([i for i in range(8, 15)], index=[str(i) for i in range(8, 15)])

pca_data = n_dims.apply(lambda n_dim: (reduce_dimensionality(data=test_data, n_final_features=n_dim), n_dim))

# Settings
print(f"[{datetime.now()}]GENERATING SETTINGS...")
EXP_NUM = 3

settings_DBSCAN = {}

print(f"[{datetime.now()}]GENERATING HYPERPARAMETERS CANDIDATES...")
param = [{'epsilon': i, 'MinPts': i} for i in range(11, 17)]

print(f"[{datetime.now()}]STARTING GridSearch...")
timestamp1 = datetime.now()

timestamp1 = datetime.now() - timestamp1
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp1}...")

print(f"[{datetime.now()}]CALCULATING SILHOUETTE SCORES...")
aux1 = pd.Series()
aux2 = pca_data.apply(lambda data: data[1])
aux2.name = 'PCA_dim'
aux3 = pd.Series()
# aux3 = aux1.apply(lambda data: silhouette_score(X=data[list(set(data.columns) - {'cluster'})], labels=data['cluster']))
aux3.name = 'silhouette'

print(f"[{datetime.now()}]PRINTING SILHOUETTE SCORES TO FILE...")
timestamp2 = datetime.now()
aux3.plot(kind="bar", xlabel="Number of dimensions after PCA", ylabel="Silhouette Score",
          figsize=(35, 30)).get_figure().savefig(
    f'Data/Results/Experiments/PCA-KMeans_sil_score{EXP_NUM}.png')
timestamp2 = datetime.now() - timestamp2
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp2}...")

print(f"[{datetime.now()}]OUTLIER DETECTION (with 8 dimensions)...")
print(f"[{datetime.now()}]{'=' * 5} DBSCAN labels {'=' * 5}")

det_dbscan = None

print(f"[{datetime.now()}]{'=' * 5}---{'=' * 5}")

# Printing log file
# Saving the reference of the standard output
original_stdout = sys.stdout
with open(f'Data/Results/Experiments/[Experiment PCA-KMeans-MATR]_main_log_{EXP_NUM}.txt', 'w') as f:
    sys.stdout = f
    # Reset the standard output
    print(f"PCA number of dimensions parameter:\t{n_dims}")
    print(f"KMeans number of cluster candidates:\t{range(10, 16)}")
    print(f"KMeans settings:\t{settings_DBSCAN}")
    print(f"Time elapsed for MATR computation (all of the datasets):\t{timestamp1}")
    print(f"Time elapsed for Silhouette Scores plotting:\t{timestamp2}")
    # print(f"Time elapsed for Clusters plotting:\t{timestamp3}")
sys.stdout = original_stdout

print(f"[{datetime.now()}]PRINTING CLUSTERING PAIRPLOTS TO FILES...")
# Printing the clusters
timestamp3 = datetime.now()
aux1.apply(lambda x: visualize_cluster(data=x,
                                       i=EXP_NUM,
                                       cluster_or_outliers='cluster',
                                       additional=f"PCA_{len(x.columns) - 1}_dim-KMEANS_{x['cluster'].max() + 1}",
                                       path="Data/Results/Experiments/"))

visualize_cluster(data=det_dbscan[list(set(det_dbscan.index) - {'cluster'} - {'LDOF'})],
                  i=EXP_NUM,
                  cluster_or_outliers='outlier',
                  additional=f"[LDOF]PCA_{len(det_dbscan.columns) - 1}_dim-KMEANS_{det_dbscan['cluster'].max() + 1}",
                  path="Data/Results/Experiments/")
timestamp3 = datetime.now() - timestamp3
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp3}...")
