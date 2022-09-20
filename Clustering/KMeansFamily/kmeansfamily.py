# K-Means Clustering prototype

# 0.Initialization:
#   a.The Forgy method randomly chooses k observations from the dataset and uses these as the initial means.
#   b.The Random Partition method first randomly assigns a cluster to each observation and then proceeds to the update step,
#     thus computing the initial mean to be the centroid of the cluster's randomly assigned points.
# 1.Given an initial set of k means m1(1),...,mk(1):
# 2.Assignment step:
#   a.Assign each observation to the cluster with the nearest mean: that with the least squared Euclidean distance.
# 3.Update step:
#   a.Recalculate means (centroids) for observations assigned to each cluster.
# 4.The algorithm has converged when the assignments no longer change.
from modin.pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


def kmeans(data: DataFrame, hyperpar: dict, settings: dict):
    """ 
    This functions assings a cluster number to each instance of data.
    :param settings: The dictionary of the algorithm's settings.
    :param hyperpar: The dictionary of the hyperparameters' values.
    :param data: This Dataframe contains the data to be clustered.
    :return The initial DataFrame, but with a new column, with a progressive id for each cluster.
    """
    fit = KMeans(n_clusters=hyperpar['n_clusters'], init='k-means++', n_init=settings['n_init'],
                 max_iter=settings['max_iter'],
                 tol=0.0001, verbose=settings['verbose'], random_state=None, copy_x=True,
                 algorithm=settings['algorithm'])
    data.loc[:, 'cluster'] = fit.fit_predict(data)
    return data


def kmedoids(data: DataFrame, hyperpar: dict, settings: dict):
    """
    This functions assings a cluster number to each instance of data.
    :param hyperpar: The dictionary of the hyperparameters' values.
    :param data: This Dataframe contains the data to be clustered.
    :param settings: This dictionary contains the settings for the optimization of the algorithm's performance.
    :return DataFrame
    """
    fit = KMedoids(n_clusters=hyperpar['n_clusters'], metric=hyperpar['metric'], init=settings['init'],
                   max_iter=settings['max_iter'], random_state=settings['random_state'])
    data.loc[:, 'cluster'] = fit.fit_predict(data)
    return data
