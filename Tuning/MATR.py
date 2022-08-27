"""
This file contains the implementation of the MATR algorithm, as described by Xinjie Fan et al. [2020].
The MATR algorithms returns the best hyperparameter lambda, given a certain clustering algorithm.
"""

# Input:
#     -clustering algorithm [A]       --> The clustering algorithm that will be used.
#     -similarity matrix [S]          --> A n x n matrix, where each cell S_ij = similarity(Data[i],Data[j]).
#     -candidates [{r1, · · · , rT} ] --> A list of possibile candidates for the number of present clusters.
#     -number of repetitions [J]      --> The number of iterations that the algorithm will execute.
#     -training ratio [y_train]       --> The proportion of the data that will be used for the training of the algorithm.
#     -trace gap [delta]              --> The margin of error of the number of clusters identified.
import numpy as np
import pandas as pd
from pandas import DataFrame, Series


def MATR(A, D: DataFrame, S: DataFrame, lambd: Series, T: int, r: int):
    """
    This function returns the ideal value for an hyperparameter lambda, by using the MATR algorithm as described by Xinjie Fan et al. [2020].
    :param A: The clustering Algorithm.
    :param D: The dataset, in the form of a DataFrame.
    :param S: The similarity matrix, in the form of a DataFrame.
    :param lambd: The candidates for the hyperparameter lambda.
    :param T:  The number of iteration to execute.
    :param r: The number of clusters present in the data.
    :return: The ideal value of the hyperparameter lambda.
    """
    L = []
    for i in range(T):
        Z = A(D, r, lambd[i])
        Z_inv = Z[i].T.dot(Z[i])
        Z_inv = pd.DataFrame(np.linalg.pinv(Z_inv.values), Z_inv.columns, Z_inv.index)
        X = Z[i].dot(Z_inv.dot(Z[i].T))
        L.append((S.dot(X), Z))
    return max(L, key=lambda x: x[0])[1]
