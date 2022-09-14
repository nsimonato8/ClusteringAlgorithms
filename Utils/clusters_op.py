from math import sqrt

from pandas import DataFrame


def freq(data: DataFrame, column: str, value) -> float:
    """
    This functions returns the freqeuncy of value in the column feature of data.

    :param data: The input dataframe
    :param column: The feature on which to calculate the frequency
    :param value: The feature of which to calculate the frequency
    :return: The frequency of value in the column column
    """
    return data[data[column] == value].count() / data.shape[0]


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
    freqss1, freqss2 = freqs1.align(freqs2, join="left", fill_value=0)
    freqss1 = freqss1.to_frame(name="cluster1")
    freqss1.loc[:, "cluster2"] = freqss2
    freqs = freqss1["cluster1"] * freqss1["cluster2"]
    # freqs = [freq(cluster1, col, cluster1[col].iloc[i]) * freq(cluster2, col, cluster1[col].iloc[i]) for i in range(cluster1.shape[0])]
    return 1. - (cluster1.shape[0] * cluster2.shape[0]) ** (-1) * sum(freqs)


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
