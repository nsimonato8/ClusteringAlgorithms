"""
This file contains the implementation of the MATR-CV algorithm, as described by Xinjie Fan et al. [2020].
The MATR-CV algorithms returns the number of clusters that are contained in the data, given a certain clustering algorithm.
"""

# Input:
#     -clustering algorithm [A]       --> The clustering algorithm that will be used.
#     -similarity matrix [S]          --> A n x n matrix, where each cell S_ij = similarity(Data[i],Data[j]).
#     -candidates [{r1, · · · , rT} ] --> A list of possibile candidates for the number of present clusters.
#     -number of repetitions [J]      --> The number of iterations that the algorithm will execute.
#     -training ratio [y_train]       --> The proportion of the data that will be used for the training of the algorithm.
#     -trace gap [delta]              --> The margin of error of the number of clusters identified.
from statistics import median

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


def nodesplitting(S: DataFrame, n: int, y_train: float):
    return S.iloc[:(n * y_train), :(n * y_train)], S.iloc[(n * y_train):, :(n * y_train)], S.iloc[(n * y_train):,
                                                                                           (n * y_train):]


def clustertest(S: DataFrame, Z: DataFrame):
    Z2 = Z.T.dot(Z)
    Z2 = pd.DataFrame(np.linalg.pinv(Z2.values), Z2.columns, Z2.index)
    M = S.dot(Z.dot(Z2))
    Z22 = pd.Dataframe()
    for i in range(S.shape[0]):
        Z22.iloc[i, M.iloc[i, :].max()] = 1  # Zˆ22(i, arg max M(i,:)) = 1
    return Z22


def matrcv(A, S: DataFrame, r: Series, J: int, y_train: float, delta: float, **settings):
    """
    This function returns the number of clusters present in the training data. It is indicated for clustering algorithms
    such as k-means and k-medoids.
    :param A:
        The clustering algorithm.
    :param S:
        The nxn similarity matrix of the data.
    :param r:
        A Pandas Series of the possible value of the number of clusters.
    :param J:
        The number of iterations that the algorithm should do.
    :param y_train:
        The proportion of data to use as training data for the algorithm.
    :param delta:
        A parameter.
    :param settings:
        The settings to use for the clustering algorithm.
    :return:
        An integer, the number of clusters present in the data.
    """
    result = []
    L = []
    for j in range(J):
        for t in range(len(r)):
            S11, S21, S22 = nodesplitting(S, S.shape[0], y_train)
            Z11 = A(S11, r.iloc[t], settings)
            Z22 = clustertest(S21, Z11)
            Z_i = Z11.T.dot(Z11)
            Z_i = pd.DataFrame(np.linalg.pinv(Z_i.values), Z_i.columns, Z_i.index)
            X22 = Z22.dot(Z_i.dot(Z22.T))
            L.append((S22, X22, r.iloc[t]))
            result.append(
                min(L, key=(lambda x: x[2] >= max(L, key=lambda i: i[2]) - delta)))  # TODO: rivedere la selezione
    return median(result)
