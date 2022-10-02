import os
import sys
import warnings
from datetime import datetime

from OutliersDetection.LDOF import top_n_LDOF

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import ray
# import pandas as pd
from scipy.spatial.distance import euclidean

from DataPreProcessing.importance import reduce_dimensionality
from Utils.Visualization.visualization import visualize_cluster

os.environ["MODIN_CPUS"] = "20"
os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray
ray.shutdown()
ray.init(num_cpus=20)

import modin.pandas as pd

FILENAME = "10k_"

master_timestamp = datetime.now()

# Importing Data
print(f"[{datetime.now()}]IMPORTING DATA...")
test_data = pd.read_csv(f"Data/{FILENAME}sessions_cleaned.csv", sep=",", skipinitialspace=True,
                        skipfooter=3)  # , engine='python')

print(f"[{datetime.now()}]REDUCING DIMENSIONALITY...")
n_dims = pd.Series([i for i in range(8, 15)], index=[str(i) for i in range(8, 15)])

pca_data = n_dims.apply(lambda n_dim: (reduce_dimensionality(data=test_data, n_final_features=n_dim), n_dim))

# Settings
print(f"[{datetime.now()}]GENERATING SETTINGS...")
EXP_NUM = 0

settings_LDOF = {
    'n': 100,
    'k': 10
}

res = pca_data['8'][0]
print(f"[{datetime.now()}]OUTLIER DETECTION (with 8 dimensions)...")
print(f"[{datetime.now()}]{'=' * 5} LDOF {'=' * 5}")
# Outlier detection JUST 8 DIMENSIONS
timestamp4 = datetime.now()

det_ldof = top_n_LDOF(data=res[list(set(res.columns) - {'cluster'})], distance=euclidean, n=settings_LDOF['n'],
                      k=settings_LDOF['k'], verbose=1)

timestamp4 = datetime.now() - timestamp4
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp4}...")
print(f"[{datetime.now()}]{'=' * 5}---{'=' * 5}")

print(f"[{datetime.now()}]PRINTING CLUSTERING PAIRPLOTS TO FILES...")
# Printing the clusters
timestamp6 = datetime.now()

visualize_cluster(data=det_ldof[list(set(det_ldof.columns) - {'cluster'} - {'LDOF'})],
                  i=EXP_NUM,
                  cluster_or_outliers='outlier',
                  additional=f"[LDOF]PCA_{len(det_ldof.columns) - 1}_dim{FILENAME}",
                  path="Data/Results/")

timestamp6 = datetime.now() - timestamp6
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp6}...")

path_results = "Data/Results/Experiments/LDOF/"

det_ldof.to_csv(path_results + "Outliers_LDOF.csv")

original_stdout = sys.stdout
with open(f'Data/Results/[Experiment PCA-KMeans-MATR]_main_log_{EXP_NUM}{FILENAME}.txt', 'w') as f:
    sys.stdout = f
    # Reset the standard output
    print(f"PCA number of dimensions parameter:\n{n_dims}\n")
    print(f"KMeans number of cluster candidates:\t{range(10, 16)}")
    print(f"LDOF settings:\t{settings_LDOF}")
    print(f"Time elapsed for Outlier Detection (LDOF):\t{timestamp4}")
sys.stdout = original_stdout
