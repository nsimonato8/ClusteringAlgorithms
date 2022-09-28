import os
import sys
import warnings
from datetime import datetime

from Clustering.Hierarchical.HAC import HAC

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import ray
# import pandas as pd
from sklearn.metrics import silhouette_score

from DataPreProcessing.importance import reduce_dimensionality
from OutliersDetection.CBOD import CBOD
from Tuning.MATR import MATR
from Utils.Visualization.visualization import visualize_cluster, plot_dendrogram

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
                        skipfooter=3)  # engine='python'

print(f"[{datetime.now()}]REDUCING DIMENSIONALITY...")
n_dims = pd.Series([i for i in range(8, 15)], index=[str(i) for i in range(8, 15)])

pca_data = n_dims.apply(lambda n_dim: (reduce_dimensionality(data=test_data, n_final_features=n_dim), n_dim))

# Settings
print(f"[{datetime.now()}]GENERATING SETTINGS...")
EXP_NUM = 1


settings_HAC = {'compute_full_tree': 'auto',
                'linkage': 'ward',
                'distance': 'euclidean',
                'epsilon': None}

settings_LDOF = {
    'n': 10,
    'k': 10
}

settings_CBOD = {
    'epsilon': 0.005
}

print(f"[{datetime.now()}]GENERATING HYPERPARAMETERS CANDIDATES...")
param = [{'n_clusters': i} for i in range(11, 17)]


print(f"[{datetime.now()}]STARTING MATR...")
timestamp1 = datetime.now()
aux1 = pca_data.apply(lambda data: MATR(A=HAC, D=data[0], hyperpar=param, settings=settings_HAC, verbose=1,
                                        path="Data/Results/Experiments/HAC/",
                                        name=f"[{EXP_NUM}]Experiment - PCA_{data[1]}_dim-HAC{FILENAME}"))
timestamp1 = datetime.now() - timestamp1
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp1}...")

print(f"[{datetime.now()}]CALCULATING SILHOUETTE SCORES...")
aux1.name = 'MATR'
aux2 = pca_data.apply(lambda data: data[1])
aux2.name = 'PCA_dim'
aux3 = aux1.apply(lambda data: silhouette_score(X=data[list(set(data.columns) - {'cluster'})], labels=data['cluster']))
aux3.name = 'silhouette'

result = pd.concat([aux2, aux3], axis=1, names=["PCA_dim", "silhouette"])

print(f"[{datetime.now()}]PRINTING SILHOUETTE SCORES TO FILE...")
timestamp2 = datetime.now()
result["silhouette"].plot(kind="bar", xlabel="Number of dimensions after PCA", ylabel="Silhouette Score",
                          figsize=(35, 30)).get_figure().savefig(
    f'Data/Results/Experiments/HAC/PCA-HAC_sil_score{EXP_NUM}{FILENAME}.png')
timestamp2 = datetime.now() - timestamp2
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp2}...")

print(f"[{datetime.now()}]OUTLIER DETECTION (with 8 dimensions)...")
res = aux1['8']

print(f"[{datetime.now()}]{'=' * 5} CBOD {'=' * 5}")
timestamp5 = datetime.now()
det_cbod = CBOD(data=res, k=res['cluster'].max() + 1, epsilon=settings_CBOD['epsilon'])

timestamp5 = datetime.now() - timestamp5
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp5}...")
print(f"[{datetime.now()}]{'=' * 5}---{'=' * 5}")

# Printing the clusters
print(f"[{datetime.now()}]PRINTING PLOTS TO FILES...")
timestamp6 = datetime.now()

aux1.apply(lambda x: plot_dendrogram(
    data=x,
    i=EXP_NUM,
    additional=f"PCA_{len(x.columns) - 1}_dim-HAC_{x['cluster'].max() + 1}{FILENAME}",
    path="Data/Results/Experiments/HAC/"))

visualize_cluster(data=res,
                  i=EXP_NUM,
                  cluster_or_outliers='cluster',
                  additional=f"PCA_{len(res.columns) - 1}_dim-KMEANS_{res['cluster'].max() + 1}{FILENAME}",
                  path="Data/Results/Experiments/HAC/")

visualize_cluster(data=det_cbod[list(set(det_cbod.columns) - {'cluster'})],
                  i=EXP_NUM,
                  cluster_or_outliers='outlier',
                  additional=f"[CBOD]PCA_{len(det_cbod.columns) - 1}_dim-HAC_{det_cbod['cluster'].max() + 1}{FILENAME}",
                  path="Data/Results/Experiments/HAC/")

timestamp6 = datetime.now() - timestamp6
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp6}...")

# Printing log file
# Saving the reference of the standard output
original_stdout = sys.stdout
with open(f'Data/Results/Experiments/HAC/[Experiment PCA-HAC-MATR]_main_log_{EXP_NUM}{FILENAME}.txt', 'w') as f:
    sys.stdout = f
    # Reset the standard output
    print(f"PCA number of dimensions parameter:\t{n_dims}")
    print(f"HAC number of cluster candidates:\t{range(10, 16)}")
    print(f"HAC settings:\t{settings_HAC}")
    print(f"LDOF settings:\t{settings_LDOF}")
    print(f"CBOD settings:\t{settings_CBOD}")
    print(f"Time elapsed for MATR computation (all of the datasets):\t{timestamp1}")
    print(f"Time elapsed for Outlier Detection (CBOD):\t{timestamp5}")
    print(f"Time elapsed for Silhouette Scores plotting:\t{timestamp2}")
    print(f"Time elapsed for Plots plotting:\t{timestamp6}")
sys.stdout = original_stdout

print(f"[{datetime.now()}]PRINTING RESULTS TO FILES...")
timestamp7 = datetime.now()

path_results = "Data/Results/Experiments/HAC/"

det_cbod.to_csv(path_results + f"HAC_Outliers_CBOD{FILENAME}.csv")

timestamp7 = datetime.now() - timestamp7
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp7}...")

master_timestamp = datetime.now() - master_timestamp
print(f"[{datetime.now()}]EXPERIMENT {EXP_NUM} CONCLUDED! Time elapsed:\t{master_timestamp}...")
