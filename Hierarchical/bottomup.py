import pandas as pd
from pandas import DataFrame


# 1.Begin with n clusters, each containing one object and we will number the clusters 1 through n.
# 2.Compute the between-cluster distance D(r, s) as the between-object distance of the two objects in r and s respectively,
#   r, s =1, 2, ..., n. Let the square matrix D = (D(r, s)). If the objects are represented by quantitative vectors,
#   we can use Euclidean distance.
# 3.Next, find the most similar pair of clusters r and s, such that the distance, D(r, s), is minimum among all the pairwise
#   distances.
# 4.Merge r and s to a new cluster t and compute the between-cluster distance D(t, k) for any existing cluster k ≠ r, s .
#   Once the distances are obtained,  delete the rows and columns corresponding to the old cluster r and s in the D matrix,
#   because r and s do not exist anymore. Then add a new row and column in D corresponding to cluster t.
# 5.Repeat Step 3 a total of n − 1 times until there is only one cluster left.
def HAC(data: DataFrame, similarity, settings: dict):
    """
    A prototype for the HAC algorithm.
    :param data: The Pandas DataFrame that contains the data to be clustered.
    :param similarity: The similarity function that will be used. It must return a float between 0 and 1.
    :param settings: Dictionary of settings that will be used for performance's optimization.
    :return: Dataframe
    """
    return pd.DataFrame()
