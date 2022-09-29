import os
import sys
import warnings
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import ray
import pandas as pd1

os.environ["MODIN_CPUS"] = "20"
os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray
ray.shutdown()
ray.init(num_cpus=20)

import modin.pandas as pd

master_timestamp = datetime.now()

print(f"[{datetime.now()}]IMPORTING DATA...")
test_data = pd.read_csv("Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3)  # engine='python'

# Main settings
EXP_NUM = 0
FILENAME = "10k_"

main_path = "Data/Results/Experiments/"

# # ---------- LDOF ----------
# print(f"[{datetime.now()}]Retrieving outliers from LDOF...")
#
# data_LDOF = pd.read_csv(main_path + "LDOF/Outliers{FILENAME}.csv")
# data_LDOF_filtered = data_LDOF.loc[data_LDOF['outlier'] == 1]
#
# # Print test_data pairplot
# print(f"\t[{datetime.now()}]Printing PairPlot LDOF...")
# outliers_LDOF = test_data.assign(outlier=data_LDOF['outlier'])
#
# kmeans_outliers_LDOF = test_data.loc[data_LDOF_filtered.index]
#
# # Print log with outliers in .txt
# print(f"\t[{datetime.now()}]Printing Log[1]...")
# original_stdout = sys.stdout
# with open(f'Data/Results/Experiments/OutliersRetrieval/Outliers_LDOF{FILENAME}.txt', 'w') as f:
#     sys.stdout = f
#     with pd1.option_context('expand_frame_repr', False, 'display.max_columns', 1000, 'display.max_rows', 10000):
#         print(f"{'=' * 5} LDOF RESULTS {'=' * 5}")
#         print(kmeans_outliers_LDOF.head(n=kmeans_outliers_LDOF.shape[0]))
#         print(f"{'=' * 5} ------------ {'=' * 5}")
#         pass
#     pass
# sys.stdout = original_stdout
#
# print(f"\t[{datetime.now()}]Printing PairPlot LDOF...")
# visualize_cluster(data=outliers_LDOF[list(set(outliers_LDOF.columns) - {'cluster'})],
#                   i=EXP_NUM,
#                   cluster_or_outliers='outlier',
#                   additional=f"[LDOF]PCA_{len(outliers_LDOF.columns) - 1}_dim{FILENAME}",
#                   path="Data/Results/Experiments/OutliersRetrieval/")

# ---------- KMEANS ----------
print(f"[{datetime.now()}]Retrieving outliers from KMEANS...")
# Read .csv with outliers data
print(f"\t[{datetime.now()}]Reading Outliers data...")

kmeans_data_CBOD = pd.read_csv(main_path + "KMEANS/KMEANS_Outliers_CBOD.csv")
kmeans_data_CBOD_filtered = kmeans_data_CBOD.loc[kmeans_data_CBOD['outlier'] == 1]

# .loc in test_data
print(f"\t[{datetime.now()}]Identifying Outliers data in original dataset...")
kmeans_outliers_CBOD = test_data.loc[kmeans_data_CBOD_filtered.index]

print(f"\t[{datetime.now()}]Printing Log...")
original_stdout = sys.stdout
with open(f'Data/Results/Experiments/OutliersRetrieval/KMeans_outliers_CBOD{FILENAME}.txt', 'w') as f:
    sys.stdout = f
    with pd1.option_context('expand_frame_repr', False, 'display.max_columns', 1000, 'display.max_rows', 10000):
        print(f"{'=' * 5} CBOD RESULTS {'=' * 5}")
        print(kmeans_outliers_CBOD.head(n=kmeans_outliers_CBOD.shape[0]))
        print(f"{'=' * 5} ------------ {'=' * 5}")
        pass
    pass
sys.stdout = original_stdout

print(f"\t[{datetime.now()}]Printing PairPlot CBOD...")
# kmeans_outliers_CBOD = test_data.assign(outlier=kmeans_data_CBOD['outlier'], cluster=kmeans_data_CBOD['cluster'])
# visualize_cluster(data=kmeans_outliers_CBOD[list(set(kmeans_outliers_CBOD.columns) - {'cluster'})],
#                   i=EXP_NUM,
#                   cluster_or_outliers='outlier',
#                   additional=f"[KMEANS]PCA_{len(kmeans_outliers_CBOD.columns) - 1}_dim-KMEANS_{kmeans_outliers_CBOD['cluster'].max() + 1}",
#                   path="Data/Results/Experiments/OutliersRetrieval/")

# ---------- HAC ----------
print(f"[{datetime.now()}]Retrieving outliers from HAC...")
# Read .csv with outliers data
print(f"\t[{datetime.now()}]Reading Outliers data...")

HAC_data_CBOD = pd.read_csv(main_path + f"HAC/HAC_Outliers_CBOD{FILENAME}.csv")
HAC_data_CBOD_filtered = HAC_data_CBOD.loc[HAC_data_CBOD['outlier'] == 1]

# .loc in test_data
print(f"\t[{datetime.now()}]Identifying Outliers data in original dataset...")
HAC_outliers_CBOD = test_data.loc[HAC_data_CBOD_filtered.index]

print(f"\t[{datetime.now()}]Printing Log[2]...")
original_stdout = sys.stdout
with open(f'Data/Results/Experiments/OutliersRetrieval/HAC_outliers_CBOD{FILENAME}.txt', 'w') as f:
    sys.stdout = f
    with pd1.option_context('expand_frame_repr', False, 'display.max_columns', 1000, 'display.max_rows', 10000):
        print(f"{'=' * 5} CBOD RESULTS {'=' * 5}")
        print(HAC_outliers_CBOD.head(n=HAC_outliers_CBOD.shape[0]))
        print(f"{'=' * 5} ------------ {'=' * 5}")
        pass
    pass
sys.stdout = original_stdout

print(f"\t[{datetime.now()}]Printing PairPlot CBOD...")
# HAC_outliers_CBOD = test_data.assign(outlier=HAC_data_CBOD['outlier'], cluster=HAC_data_CBOD['cluster'])
# visualize_cluster(data=HAC_outliers_CBOD[list(set(HAC_outliers_CBOD.columns) - {'cluster'})],
#                   i=EXP_NUM,
#                   cluster_or_outliers='outlier',
#                   additional=f"[HAC]PCA_{len(HAC_outliers_CBOD.columns) - 1}_dim-HAC_{HAC_outliers_CBOD['cluster'].max() + 1}{FILENAME}",
#                   path="Data/Results/Experiments/OutliersRetrieval/")

# # ---------- DBSCAN ----------
# print(f"[{datetime.now()}]Retrieving outliers from DBSCAN...")
# # Read .csv with outliers data
# print(f"\t[{datetime.now()}]Reading Outliers data...")
# DBSCAN_data_filtering = pd.read_csv(main_path + "DBSCAN/DBSCAN_Outliers.csv")
#
# # .loc in test_data
# print(f"\t[{datetime.now()}]Identifying Outliers data in original dataset...")
# DBSCAN_outliers_filtering = test_data.loc[DBSCAN_data_filtering.index]
#
# # Print log with outliers in .txt
# print(f"\t[{datetime.now()}]Printing Log...")
# original_stdout = sys.stdout
# with open(f'Data/Results/DBSCAN_outliers_filtering.txt', 'w') as f:
#     sys.stdout = f
#     with pd1.option_context('expand_frame_repr', False, 'display.max_columns', 1000, 'display.max_rows', 10000):
#         print(f"{'=' * 5} DBSCAN RESULTS {'=' * 5}")
#         print(DBSCAN_outliers_filtering.head(n=DBSCAN_outliers_filtering.shape[0]))
#         print(f"{'=' * 5} ------------ {'=' * 5}")
#         pass
#     pass
# sys.stdout = original_stdout
#
# # Print test_data pairplot
# print(f"\t[{datetime.now()}]Printing PairPlot Filtering...")
# DBSCAN_outliers_filtering = test_data.assign(outlier=DBSCAN_data_filtering['outlier'],
#                                              cluster=DBSCAN_data_filtering['cluster'])
# visualize_cluster(data=DBSCAN_outliers_filtering[list(set(DBSCAN_outliers_filtering.columns) - {'cluster'})],
#                   i=EXP_NUM,
#                   cluster_or_outliers='outlier',
#                   additional=f"[DBSCAN]PCA_{len(DBSCAN_outliers_filtering.columns) - 1}_dim-DBSCAN_{DBSCAN_outliers_filtering['cluster'].max() + 1}",
#                   path="Data/Results/Experiments/OutliersRetrieval/")

master_timestamp = datetime.now() - master_timestamp
print(f"[{datetime.now()}]EXPERIMENT {EXP_NUM} CONCLUDED! Time elapsed:\t{master_timestamp}...")
