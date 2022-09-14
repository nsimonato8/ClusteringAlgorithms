"""
Implementation of the CBOD algorithm, as described by Sheng-yi Jiang et al. [2008]
"""

# Stage 1. Clustering: Cluster on data set D and produce clustering results C ={C1,C2 ,",Ck} .
# Stage 2. Determining Outlier Clusters:
#     Compute outlier factor OF(C_i)(1 <= i <= k) , sort clustersC = {C1,C2 ,..,Ck} according to their OF.
#     Search the minimum b , which satisfies sum(from=0, to=b, |C_i|)/|D| >= epsilon, with 0< = epsilon <= 1
#     Label clusters C_1, C_2, ..., C_b  with ‘outlier’
#     Label clusters C_b+1, C_b+2, ..., C_k  with ‘normal’
from pandas import DataFrame

from Utils.clusters_op import cluster_distance_CBOD


def outlier_factor(cluster: DataFrame, data: DataFrame, k: int) -> float:
    """
    This function returns the outlier factor of a given cluster, as described by Sheng-yi Jiang et al. [2008]

    :param cluster: The input cluster
    :param data: The clustered dataset
    :param k: The number of clusters
    :return: The outlier factor OF, as described by Sheng-yi Jiang et al. [2008]
    """

    cluster_set = [data[data['cluster'] == i] for i in range(k)]
    cluster_set = filter(lambda x: not x.equals(cluster), cluster_set)

    return sum([cluster_distance_CBOD(cluster, i) * i.shape[0] for i in cluster_set])


def CBOD(data: DataFrame, k: int, epsilon: float) -> None:
    """
    This function labels the instances of the dataset as 'outlier' or 'normal' according to the CBOD algorithm,
    as described by Sheng-yi Jiang et al. [2008]

    :param epsilon: An approximate ratio of the outliers to the whole training dataset.
    :param data: The clustered DataFrame
    :param k: The number of clusters
    :return: None
    """
    clusters = [data[data['cluster'] == i] for i in range(k)]
    clusters_repr = [outlier_factor(cluster=clusters[i], data=data, k=k) for i in range(k)]
    clusters_repr.sort(reverse=True)
    b = 0
    while b < k and (sum(map(lambda x: x.shape[0], clusters[0:b])) / data.shape[0]) > epsilon:
        b += 1

    data.loc[:, "outlier"] = data['cluster'].apply(lambda x: True if x <= b else False, axis=1)
    pass
