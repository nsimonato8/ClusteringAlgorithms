import os
import sys
import warnings
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import ray
import pandas as pd1

from Utils.Visualization.visualization import visualize_cluster

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
main_path = "Data/Results/Experiments/"

# ---------- KMEANS ----------
print(f"[{datetime.now()}]Retrieving outliers from KMEANS...")
# Read .csv with outliers data
kmeans_data_LDOF = pd.read_csv(main_path + "KMEANS/KMEANS_Outliers_LDOF.csv")
kmeans_data_CBOD = pd.read_csv(main_path + "KMEANS/KMEANS_Outliers_CBOD.csv")

# .loc in test_data
kmeans_outliers_LDOF = test_data.loc[kmeans_data_LDOF.index]
kmeans_outliers_CBOD = test_data.loc[kmeans_data_CBOD.index]

# Print log with outliers in .txt
original_stdout = sys.stdout
with open(f'Data/Results/KMeans_outliers_LDOF.txt', 'w') as f:
    sys.stdout = f
    with pd1.option_context('expand_frame_repr', False):
        print(f"{'=' * 5} LDOF RESULTS {'=' * 5}")
        print(kmeans_outliers_LDOF.head(n=kmeans_outliers_LDOF.shape[0]))
        print(f"{'=' * 5} ------------ {'=' * 5}")
        pass
    pass
sys.stdout = original_stdout

original_stdout = sys.stdout
with open(f'Data/Results/KMeans_outliers_CBOD.txt', 'w') as f:
    sys.stdout = f
    with pd1.option_context('expand_frame_repr', False):
        print(f"{'=' * 5} CBOD RESULTS {'=' * 5}")
        print(kmeans_outliers_CBOD.head(n=kmeans_outliers_CBOD.shape[0]))
        print(f"{'=' * 5} ------------ {'=' * 5}")
        pass
    pass
sys.stdout = original_stdout

# Print test_data pairplot
kmeans_outliers_LDOF = test_data.assign(outlier=kmeans_data_LDOF['outlier'], cluster=kmeans_data_LDOF['cluster'])
visualize_cluster(data=kmeans_outliers_LDOF[list(set(kmeans_outliers_LDOF.columns) - {'cluster'})],
                  i=EXP_NUM,
                  cluster_or_outliers='outlier',
                  additional=f"[DBSCAN]PCA_{len(kmeans_outliers_LDOF.columns) - 1}_dim-DBSCAN_{kmeans_outliers_LDOF['cluster'].max() + 1}",
                  path="Data/Results/Experiments/DBSCAN/")

kmeans_outliers_CBOD = test_data.assign(outlier=kmeans_data_CBOD['outlier'], cluster=kmeans_data_CBOD['cluster'])
visualize_cluster(data=kmeans_outliers_CBOD[list(set(kmeans_outliers_CBOD.columns) - {'cluster'})],
                  i=EXP_NUM,
                  cluster_or_outliers='outlier',
                  additional=f"[DBSCAN]PCA_{len(kmeans_outliers_CBOD.columns) - 1}_dim-DBSCAN_{kmeans_outliers_CBOD['cluster'].max() + 1}",
                  path="Data/Results/Experiments/DBSCAN/")

# ---------- HAC ----------
print(f"[{datetime.now()}]Retrieving outliers from HAC...")
# Read .csv with outliers data
HAC_data_LDOF = pd.read_csv(main_path + "HAC/HAC_Outliers_LDOF.csv")
HAC_data_CBOD = pd.read_csv(main_path + "HAC/HAC_Outliers_CBOD.csv")

# .loc in test_data
HAC_outliers_LDOF = test_data.loc[HAC_data_LDOF.index]
HAC_outliers_CBOD = test_data.loc[HAC_data_CBOD.index]

# Print log with outliers in .txt
original_stdout = sys.stdout
with open(f'Data/Results/HAC_outliers_LDOF.txt', 'w') as f:
    sys.stdout = f
    with pd1.option_context('expand_frame_repr', False):
        print(f"{'=' * 5} LDOF RESULTS {'=' * 5}")
        print(HAC_outliers_LDOF.head(n=HAC_outliers_LDOF.shape[0]))
        print(f"{'=' * 5} ------------ {'=' * 5}")
        pass
    pass
sys.stdout = original_stdout

original_stdout = sys.stdout
with open(f'Data/Results/HAC_outliers_CBOD.txt', 'w') as f:
    sys.stdout = f
    with pd1.option_context('expand_frame_repr', False):
        print(f"{'=' * 5} CBOD RESULTS {'=' * 5}")
        print(HAC_outliers_CBOD.head(n=HAC_outliers_CBOD.shape[0]))
        print(f"{'=' * 5} ------------ {'=' * 5}")
        pass
    pass
sys.stdout = original_stdout

# Print test_data pairplot
HAC_outliers_LDOF = test_data.assign(outlier=HAC_data_LDOF['outlier'], cluster=HAC_data_LDOF['cluster'])
visualize_cluster(data=HAC_outliers_LDOF[list(set(HAC_outliers_LDOF.columns) - {'cluster'})],
                  i=EXP_NUM,
                  cluster_or_outliers='outlier',
                  additional=f"[DBSCAN]PCA_{len(HAC_outliers_LDOF.columns) - 1}_dim-DBSCAN_{HAC_outliers_LDOF['cluster'].max() + 1}",
                  path="Data/Results/Experiments/DBSCAN/")

HAC_outliers_CBOD = test_data.assign(outlier=HAC_data_CBOD['outlier'], cluster=HAC_data_CBOD['cluster'])
visualize_cluster(data=HAC_outliers_CBOD[list(set(HAC_outliers_CBOD.columns) - {'cluster'})],
                  i=EXP_NUM,
                  cluster_or_outliers='outlier',
                  additional=f"[DBSCAN]PCA_{len(HAC_outliers_CBOD.columns) - 1}_dim-DBSCAN_{HAC_outliers_CBOD['cluster'].max() + 1}",
                  path="Data/Results/Experiments/DBSCAN/")

# ---------- DBSCAN ----------
print(f"[{datetime.now()}]Retrieving outliers from DBSCAN...")
# Read .csv with outliers data
DBSCAN_data_filtering = pd.read_csv(main_path + "DBSCAN/DBSCAN_Outliers.csv")

# .loc in test_data
DBSCAN_outliers_filtering = test_data.loc[DBSCAN_data_filtering.index]

# Print log with outliers in .txt
original_stdout = sys.stdout
with open(f'Data/Results/DBSCAN_outliers_filtering.txt', 'w') as f:
    sys.stdout = f
    with pd1.option_context('expand_frame_repr', False):
        print(f"{'=' * 5} DBSCAN RESULTS {'=' * 5}")
        print(DBSCAN_outliers_filtering.head(n=DBSCAN_outliers_filtering.shape[0]))
        print(f"{'=' * 5} ------------ {'=' * 5}")
        pass
    pass
sys.stdout = original_stdout

# Print test_data pairplot
DBSCAN_outliers_filtering = test_data.assign(outlier=DBSCAN_data_filtering['outlier'],
                                             cluster=DBSCAN_data_filtering['cluster'])
visualize_cluster(data=DBSCAN_outliers_filtering[list(set(DBSCAN_outliers_filtering.columns) - {'cluster'})],
                  i=EXP_NUM,
                  cluster_or_outliers='outlier',
                  additional=f"[DBSCAN]PCA_{len(DBSCAN_outliers_filtering.columns) - 1}_dim-DBSCAN_{DBSCAN_outliers_filtering['cluster'].max() + 1}",
                  path="Data/Results/Experiments/DBSCAN/")

master_timestamp = datetime.now() - master_timestamp
print(f"[{datetime.now()}]EXPERIMENT {EXP_NUM} CONCLUDED! Time elapsed:\t{master_timestamp}...")