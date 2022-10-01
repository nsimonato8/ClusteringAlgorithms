import os
import sys
import warnings
from datetime import datetime

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import ray
# import pandas as pd

from Utils.Visualization.visualization import visualize_cluster

os.environ["MODIN_CPUS"] = "20"
os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray
ray.shutdown()
ray.init(num_cpus=20)

import modin.pandas as pd

master_timestamp = datetime.now()
EXP_NUM = 4
FILENAME = ""
# Importing Data
print(f"[{datetime.now()}]IMPORTING DATA...")
test_data = pd.read_csv(f"Data/{FILENAME}sessions_cleaned.csv", sep=",", skipinitialspace=True,
                        skipfooter=3)  # engine='python'

print(f"[{datetime.now()}]REDUCING DIMENSIONALITY...")
# n_dims = pd.Series([i for i in range(8, 15)], index=[str(i) for i in range(8, 15)])
#
# pca_data = n_dims.apply(lambda n_dim: (reduce_dimensionality(data=test_data, n_final_features=n_dim), n_dim))

# Settings
print(f"[{datetime.now()}]GENERATING SETTINGS...")

settings_GridSearch = {'estimator': DBSCAN(),
                       'n_jobs': -1,
                       'refit': True,
                       'verbose': 3,
                       'return_train_score': True,
                       'scoring': silhouette_score
                       }
settings_DBSCAN = {'eps': [x for x in np.arange(0., 10., 0.001)],
                   'min_samples': [x for x in range(0, 1000, 1)],
                   'metric': [euclidean],
                   'algorithm': ['auto'],
                   'n_jobs': [-1]}

print(f"[{datetime.now()}]GENERATING HYPERPARAMETERS CANDIDATES...")

print(f"[{datetime.now()}]STARTING GridSearch...")
timestamp1 = datetime.now()
model = GridSearchCV(estimator=settings_GridSearch['estimator'],
                     n_jobs=settings_GridSearch['n_jobs'],
                     refit=settings_GridSearch['refit'],
                     verbose=settings_GridSearch['verbose'],
                     return_train_score=settings_GridSearch['return_train_score'],
                     scoring=settings_GridSearch['scoring'],
                     param_grid=settings_DBSCAN)


def fit_predict(unfitted_model: GridSearchCV, X: pd.DataFrame):
    unfitted_model.fit(X)
    return unfitted_model.predict(X)


# pca_labels = pca_data.apply(lambda x: (x, fit_predict(model, x)))
#
# aux1 = pca_data.apply(lambda x: x[0].assign(cluster=x[1]))
aux1 = test_data.apply(lambda x: (x, fit_predict(model, x)))
aux1.name = 'clustered'

timestamp1 = datetime.now() - timestamp1
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp1}...")

print(f"[{datetime.now()}]CALCULATING SILHOUETTE SCORES...")
timestamp2 = datetime.now()

# aux2 = aux1.apply(lambda data: silhouette_score(X=data[list(set(data.columns) - {'cluster'})], labels=data['cluster']))
# aux2.name = 'silhouette'

timestamp2 = datetime.now() - timestamp2

print(f"[{datetime.now()}]PRINTING SILHOUETTE SCORES TO FILE...")

# aux2.plot(kind="bar", xlabel="Number of dimensions after PCA", ylabel="Silhouette Score",
#           figsize=(35, 30)).get_figure().savefig(
#     f'Data/Results/Experiments/DBSCAN/PCA-DBSCAN_sil_score{EXP_NUM}.png')

print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp2}...")

print(f"[{datetime.now()}]OUTLIER DETECTION (with 8 dimensions)...")
print(f"[{datetime.now()}]{'=' * 5} DBSCAN labels {'=' * 5}")

# det_dbscan = aux1['8'].assign(outlier=aux1['8']['cluster'].apply(lambda x: 1 if x < 0 else 0))
det_dbscan = aux1.assign(outlier=aux1['cluster'].apply(lambda x: 1 if x < 0 else 0))

print(f"[{datetime.now()}]{'=' * 5}---{'=' * 5}")

print(f"[{datetime.now()}]PRINTING CLUSTERING PAIRPLOTS TO FILES...")
# Printing the clusters
timestamp3 = datetime.now()

print(f"\t[{datetime.now()}]\tCLUSTER PLOT...")
# visualize_cluster(data=aux1['8'],
#                   i=EXP_NUM,
#                   cluster_or_outliers='cluster',
#                   additional=f"PCA_{len(aux1['8'].columns) - 1}_dim-DBSCAN_{aux1['8']['cluster'].max() + 1}_{FILENAME}{EXP_NUM}",
#                   path="Data/Results/Experiments/DBSCAN/")
visualize_cluster(data=aux1,
                  i=EXP_NUM,
                  cluster_or_outliers='cluster',
                  additional=f"PCA_{len(aux1.columns) - 1}_dim-DBSCAN_{aux1['cluster'].max() + 1}_{FILENAME}{EXP_NUM}",
                  path="Data/Results/Experiments/DBSCAN/")

print(f"\t[{datetime.now()}]\tOUTLIER PLOT...")
# visualize_cluster(data=det_dbscan[list(set(det_dbscan.columns) - {'cluster'})],
#                   i=EXP_NUM,
#                   cluster_or_outliers='outlier',
#                   additional=f"[DBSCAN]PCA_{len(det_dbscan.columns) - 1}_dim-DBSCAN_{det_dbscan['cluster'].max() + 1}_{FILENAME}{EXP_NUM}",
#                   path="Data/Results/Experiments/DBSCAN/")
visualize_cluster(data=det_dbscan[list(set(det_dbscan.columns) - {'cluster'})],
                  i=EXP_NUM,
                  cluster_or_outliers='outlier',
                  additional=f"[DBSCAN]PCA_{len(det_dbscan.columns) - 1}_dim-DBSCAN_{det_dbscan['cluster'].max() + 1}_{FILENAME}{EXP_NUM}",
                  path="Data/Results/Experiments/DBSCAN/")

timestamp3 = datetime.now() - timestamp3
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp3}...")

original_stdout = sys.stdout
with open(f'Data/Results/Experiments/DBSCAN/[Experiment PCA-DBSCAN-GridSearchCV]{FILENAME}_main_log_{EXP_NUM}.txt',
          'w') as f:
    sys.stdout = f
    # print(f"PCA number of dimensions parameter:\t{n_dims}")
    print(f"GridSearchCV settings:\t{settings_GridSearch}")
    print(f"DBSCAN settings:\t{settings_DBSCAN}")
    print(f"Time elapsed for GridSearchCV computation (all of the datasets):\t{timestamp1}")
    print(f"Time elapsed for Plotting:\t{timestamp3}")
sys.stdout = original_stdout

print(f"[{datetime.now()}]PRINTING RESULTS TO FILES...")
timestamp7 = datetime.now()

path_results = "Data/Results/Experiments/DBSCAN/"

det_dbscan.to_csv(path_results + f"DBSCAN_Outliers_Filtering{FILENAME}{EXP_NUM}.csv")

timestamp7 = datetime.now() - timestamp7
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp7}...")

master_timestamp = datetime.now() - master_timestamp
print(f"[{datetime.now()}]EXPERIMENT {EXP_NUM} CONCLUDED! Time elapsed:\t{master_timestamp}...")
