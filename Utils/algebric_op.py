import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.spatial.distance import pdist, squareform


def similarity_matrix(data: DataFrame, similarity):
    """
    This function calculates the similarity matrix of a given dataset.
    :param data: The input dataset.
    :param similarity: The similarity measure used.
    :return: The similarity matrix as a Pandas DataFrame
    """
    # result = []
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[0]):
    #         result.append(similarity(data.loc[i, :].squeeze(), data.loc[j, :].squeeze()))
    #     return pd.DataFrame(result, columns=data.columns)
    dists = pdist(data, similarity)
    return pd.DataFrame(squareform(dists), columns=data.index, index=data.index)


def inverse_matrix(data: DataFrame):
    """
    This function calculates the inverse of the matrix given in input.
    :param data: The matrix to inverse.
    :return: The inverted matrix.
    """
    return pd.DataFrame(np.linalg.pinv(data.values), data.columns, data.index)


def trace(mat: DataFrame):
    """
    This function calculates the trace of a given symmetric matrix.
    :param mat: The input matrix.
    :return: The trace of the input matrix.
    """
    assert mat.shape[0] == mat.shape[1], "The matrix is not symmetric"
    return sum([mat.iloc[i, i] for i in range(mat.shape[0])])
