"""
The algorithm computes LDOF factor on a data instance which indicate how much it deviates from its neighborhood.
Data instances obtaining high scores are more likely considered as outliers.
LDOF factor is calculated by dividing the KNN distance of an object xp by the KNN inner distance of an object xp.
This file presents an implementation of the LDOF algorithm, as described by Abir Smiti[2020].
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from modin.pandas import DataFrame, Series
from Utils.algebric_op import similarity_matrix, trace


# from pandas import DataFrame, Series


def not_col(x: Series, col: str) -> Series:
    return x[list(set(x.index) - set(col))]


def p_neighbourhood(p: Series, data: DataFrame, k: int, distance: callable) -> DataFrame:
    """
    This function retrieves the k nearest neighbours of the p instance, contained in the data DataFrame.

    :param p: The referenced instance.
    :param data: The whole dataset
    :param k: The number of neighbours to retrieve
    :param distance: The distance function to use
    :return: The k-Nearest-Neighbours
    """
    data['d'] = data.apply(lambda x: distance(p, not_col(x, "d")), axis=1)
    return data.sort_values(axis=0, by="d", ascending=True).head(n=k + 1)


def kNN_distance(p: Series, data: DataFrame, k: int, distance: callable) -> float:
    """
    This function returns the kNN distance, as described by Abir Smiti[2020].

    :param p: The referenced instance.
    :param data: The whole dataset
    :param k: The number of neighbours to retrieve
    :param distance: The distance function to use
    :return: The k-Nearest-Neighbours distance.
    """
    kNN = p_neighbourhood(p, data, k, distance)
    return kNN['d'].sum() / k


def kNN_inner_distance(data: DataFrame, k: int, distance: callable) -> float:
    """
    This function returns the kNN inner distance, as described by Abir Smiti[2020].

    :param data: The whole dataset
    :param k: The number of neighbours to retrieve
    :param distance: The distance function to use
    :return: The k-Nearest-Neighbours inner distance.
    """
    sim = similarity_matrix(data, distance)
    return (sim.values.sum() - trace(sim)) / k / (k - 1)


def LDOF_score(p: Series, data: DataFrame, k: int, distance: callable) -> float:
    """
    This function returns the LDOF score of a given instance, as described by Abir Smiti[2020].

    :param p: The referenced instance.
    :param data: The whole dataset
    :param k: The number of neighbours to retrieve
    :param distance: The distance function to use
    :return: The k-Nearest-Neighbours inner distance.
    """
    return kNN_distance(p, data, k, distance) / kNN_inner_distance(data, k, distance)


def top_n_LDOF(data: DataFrame, distance: callable, n: int, k: int) -> DataFrame:
    """
    This function implements the Top-n LDOF algorithm, as described by Abir Smiti[2020].

    :param n: The number of outliers to retrieve
    :param data: The whole dataset
    :param k: The number of neighbours to retrieve
    :param distance: The distance function to use
    :return: The k-Nearest-Neighbours inner distance.
    """
    data['LDOF'] = data.apply(lambda x: LDOF_score(x, data, k, distance), axis=1)
    data = data.sort_values(axis=0, by="d", ascending=False)
    data.drop(["LDOF"], axis=1, inplace=True)
    data['outlier'] = True
    data['outlier'].iloc[(n + 1):] = False
    return data
