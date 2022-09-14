from math import sqrt

from pandas import DataFrame


def diff_CBOD(cluster1: DataFrame, cluster2: DataFrame, col: str) -> float:
    """
    This function returns the difference between two clusters on a given feature, as described by Sheng-yi Jiang et al. [2008]

    :param cluster1: The first cluster
    :param cluster2: The second cluster
    :param col: The feature on which to compute the difference
    :return: The difference between the two clusters
    """
    freqs1 = [(cluster1[cluster1[col] == i].count() / cluster1.shape[0]) for i in cluster1[col]]
    freqs2 = [(cluster2[cluster2[col] == i].count() / cluster2.shape[0]) for i in cluster1[col]]
    freqs = sum(map(lambda x1, x2: x1 * x2, freqs1, freqs2))
    return 1. - (cluster1.shape[0] * cluster2.shape[0]) ** (-1) * sum(freqs)


def cluster_distance_CBOD(cluster1: DataFrame, cluster2: DataFrame) -> float:
    """
    The global distance between two clusters, as described by Sheng-yi Jiang et al. [2008]

    :param cluster1: The first cluster
    :param cluster2: The second cluster
    :return: The global distance between the two clusters
    """
    assert cluster1 is not None and cluster2 is not None, "cluster1 and cluster2 must not be None"
    return sqrt(sum([diff_CBOD(cluster1, cluster2, i) / len(cluster1.columns) for i in cluster1.columns]))
