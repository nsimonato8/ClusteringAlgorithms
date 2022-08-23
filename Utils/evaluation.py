from statistics import mean

from pandas import DataFrame


# Intrinsic evaluation
# 1. The mean distance between each instance and its cluster is calculated [a(o), with o the instance]
# 2. The mean instance between each instance and every other instance in a different cluster is calculated [b(o)]
# 3. The silhouette score is calculated [ (b(o) - a(o)) / max{a(o), b(o)} ]
# 4. For each cluster, the mean silhouette score is calculated.
# 5. The mean score among the clusters is returned.
def silhouette_score(data: DataFrame, distance):
    """
        This functions calculates a score of the clustering by using the silhouette score. More information at https://en.wikipedia.org/wiki/Silhouette_(clustering)
    :param data:
        A Pandas DataFrame that must contain exactly a column named "cluster", that indicates the id of the instance's cluster.
    :param distance:
        The distance function used.
    :return:
        A number between -1 and 1 that represents the score of the clustering.
    """
    assert data['cluster'].size > 0, f"Column 'cluster' does not exists."

    k = data['cluster'].max()
    clusters_mean_index = [1] * k

    for i in range(k):
        cluster = data.loc[data['cluster'] == i + 1].copy()
        out_cluster = data.loc[data['cluster'] != i + 1].copy()
        cluster = a_index(cluster, distance)
        cluster = b_index(cluster, out_cluster, distance)
        cluster['max_index'] = cluster.loc[:, ['a_index', 'b_index']].max(axis=1)
        cluster['silhouette_index'] = (cluster['b_index'] - cluster['a_index']) / cluster['max_index']
        clusters_mean_index[i] = cluster['silhouette_index'].mean()

    return mean(clusters_mean_index)


def a_index(cluster: DataFrame, distance):
    """
    Adds the column 'a_index' to the cluster passed in input.

    The a_index is the mean distance between an element of the cluster and every other element in the cluster.
    :param cluster:
        The cluster in which the a_index is calculated.
    :param distance:
        The distance function used.
    :return:
        The cluster passed in input, but with the column 'a_index' added.
    """
    cluster['a_index'] = cluster['cluster'].copy()
    for i in range(cluster.shape[0]):
        result = 0
        for j in range(cluster.shape[0]):
            if i != j:
                result += distance(cluster.iloc[i], cluster.iloc[j])
        cluster.iloc[i, cluster.columns.get_loc("a_index")] = result / cluster.shape[0]
    return cluster


def b_index(in_cluster: DataFrame, out_cluster: DataFrame, distance):
    """
    Adds the column 'b_index' to the cluster passed in input. The b_index is the mean distance between an element of the in_cluster
    and every other element in the out_cluster.
    :param in_cluster:
        The cluster in which the b_index is calculated.
    :param out_cluster:
        The dataframe that contains each element that does not belong to the in_cluster
    :param distance:
        The distance function used.
    :return:
        The cluster passed in input, but with the column 'b_index' added.
    """
    in_cluster['b_index'] = in_cluster['cluster'].copy()
    for i in range(in_cluster.shape[0]):
        result = 0
        for j in range(out_cluster.shape[0]):
            result += distance(in_cluster.iloc[i], out_cluster.iloc[j])
        in_cluster.iloc[i, in_cluster.columns.get_loc("b_index")] = result / out_cluster.shape[0]
    return in_cluster
