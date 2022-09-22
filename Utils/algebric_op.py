import modin.pandas as pd
import numpy as np
from modin.pandas import DataFrame
# import pandas as pd
# from pandas import DataFrame
from scipy.spatial.distance import pdist, squareform


def similarity_matrix(data: DataFrame, similarity: callable) -> DataFrame:
    """
    This function calculates the similarity matrix of a given dataset.

    :param data: The input dataset.
    :param similarity: The similarity measure used.
    :return: The similarity matrix as a Pandas DataFrame
    """
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
    return pd.Series([mat.iloc[i, i] for i in range(mat.shape[0])]).sum()


def construct_clustering_matrix(data: DataFrame):
    Z = pd.DataFrame(np.zeros(shape=(data.shape[0], (data['cluster'].max() + 2))))
    for i in range(data.shape[0]):
        Z.iloc[i, data['cluster'].iloc[i] + 1] = 1
    return Z
