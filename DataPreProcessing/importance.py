import modin.pandas as pd
from modin.pandas import DataFrame
from sklearn.decomposition import PCA


def reduce_dimensionality(data: DataFrame, n_final_features: int = 10) -> DataFrame:
    """
    This function applies the Scikit-Learn implementation of the PCA algorithm.
    :param data: The Pandas DataFrame to reduce.
    :param n_final_features: The number of feature to keep.
    :return: The transformed DataFrame
    """
    assert data.shape[
               1] > n_final_features, f"[starting]: {data.shape[1]} | [final]: {n_final_features}\nThe final number of feature must be strictly inferior of the starting number of features."
    model = PCA(n_components=n_final_features)
    return pd.DataFrame(model.fit_transform(data))
