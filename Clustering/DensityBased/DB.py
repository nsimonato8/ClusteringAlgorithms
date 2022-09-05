from pandas import DataFrame
from sklearn.cluster import DBSCAN


def dbscan(data: DataFrame, hyperpar: dict):
    """
    Implementation of the DB-Scan algorithm. For more information, see: https://en.wikipedia.org/wiki/DBSCAN
    :param hyperpar: the dictionary that contains the hyperparameters' values.
    :param data:  The input DataFrame.
    :return: DataFrame
    """
    fit = DBSCAN(eps=hyperpar['epsilon'], min_samples=hyperpar['minpts'], metric=hyperpar[
        'similarity'])  # n_jobs=None This parameter sets the number of processors that can be used
    data.loc[:, 'cluster'] = fit.fit_predict(data)
    return data
