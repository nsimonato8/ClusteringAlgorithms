from math import sqrt

import modin.pandas as pd
from modin.pandas import DataFrame


# from pandas import DataFrame


def diff_CBOD(cluster1: DataFrame, cluster2: DataFrame, col: str) -> float:
    """
    This function returns the difference between two clusters on a given feature, as described by Sheng-yi Jiang et al. [2008]

    :param cluster1: The first cluster
    :param cluster2: The second cluster
    :param col: The feature on which to compute the difference
    :return: The difference between the two clusters
    """
    freqs1 = cluster1[col].value_counts(normalize=False)
    freqs2 = cluster2[col].value_counts(normalize=False)
    freqss1, freqss2 = freqs1.align(freqs2, join="left", fill_value=0, axis=0)
    freqss = pd.DataFrame([freqss1, freqss2], columns=["cluster1", "cluster2"])
    print(f"{freqss.head(n=5)}")
    freqs = freqss.apply(lambda x: x["cluster1"] * x["cluster2"])
    # freqs = [freq(cluster1, col, cluster1[col].iloc[i]) * freq(cluster2, col, cluster1[col].iloc[i]) for i in range(cluster1.shape[0])]
    return 1. - (cluster1.shape[0] * cluster2.shape[0]) ** (-1) * freqs.sum()


def cluster_distance_CBOD(cluster1: DataFrame, cluster2: DataFrame) -> float:
    """
    The global distance between two clusters, as described by Sheng-yi Jiang et al. [2008]

    :param cluster1: The first cluster
    :param cluster2: The second cluster
    :return: The global distance between the two clusters
    """
    assert cluster1 is not None and cluster2 is not None, "cluster1 and cluster2 must not be None"
    assert cluster1.shape[0] > 0 and cluster1.shape[1] > 0, "cluster1 is empty"
    assert cluster2.shape[0] > 0 and cluster2.shape[1] > 0, "cluster2 is empty"
    return sqrt(sum([diff_CBOD(cluster1, cluster2, i) / cluster1.shape[1] for i in cluster1.columns]))
