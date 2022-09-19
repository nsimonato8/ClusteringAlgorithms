from math import sqrt

import numpy as np
from modin.pandas import Series
from scipy.spatial.distance import euclidean


def euclidean_distance(x, y):
    """
    Simple function that returns the euclidean distance beetween two vectors.
    :param x: Vector 1.
    :param y: Vector 2.
    :return: The euclidean distance beetween Vector 1 and Vector 2. For more info, see:  https://en.wikipedia.org/wiki/Euclidean_distance
    """
    if isinstance(x, Series) and isinstance(y, Series):
        if x.empty or y.empty:
            return float(not x.empty and not y.empty)
        return sqrt((x ** 2 + y ** 2).sum(axis=0))
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return euclidean(x, y)

    raise ValueError("X or Y are neither Pandas Series nor NumPy ndarrays")
    pass


def cosine_distance(x: Series, y: Series):
    """
    Simple function that returns the cosine similarity beetween two vectors.
    :param x: Vector 1.
    :param y: Vector 2.
    :return: The cosine similarity beetween Vector 1 and Vector 2. For more info, see: https://en.wikipedia.org/wiki/Cosine_similarity
    """
    to_exclude = ["cluster"]
    if x.empty or y.empty:
        return float(not x.empty and not y.empty)
    x = x[list(set(x.index) - set(to_exclude))]
    y = y[list(set(y.index) - set(to_exclude))]
    return x.dot(y) / sqrt((x ** 2).sum()) / sqrt((y ** 2).sum())


def minkowski_distance(x: Series, y: Series, p: int = 2):
    """
    Simple function that returns the Minkowski distance beetween two vectors.
    :param p: The order of the Minkowski distance.
    :param x: Vector 1.
    :param y: Vector 2.
    :return: The Minkowski distance beetween Vector 1 and Vector 2. For more info, see: https://en.wikipedia.org/wiki/Minkowski_distance
    """
    to_exclude = ["cluster"]
    if x.empty or y.empty:
        return float(not x.empty and not y.empty)
    x = x[list(set(x.index) - set(to_exclude))]
    y = y[list(set(y.index) - set(to_exclude))]
    return ((x - y).abs() ** p).sum() ** (1 / p)
