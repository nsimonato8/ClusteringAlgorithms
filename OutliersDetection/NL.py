"""
This file contains an implementation of the NL algorithm, as proposed by Knorr et al. [1998]
The NL algorithm identifies the outliers present in the input data.
"""

# 1. Fill the first array (of size B% of the dataset) with a block of tuples from T.
# 2. For each tuple t_i in the first array, do:
#   a. count_i <- 0
#   b. For each tuple t_j in the second array:
#       if dist(t_i, t_j) <= D:
#           Increment count_i by 1.
#       If count_i > M, mark t_i as a non-outlier and proceed to next ti.
# 3. While blocks remain to be compared to the first array, do:
#   a. Fill the second array with another block (but save a block which has never served as the first array, for last).
#   b. For each unmarked tuple t; in the first array do:
#       For each tuple tj in the second array, if dist(ti, tj) <= D:
#           Increment count_i by 1.
#           If counti > M, mark ti as a non-outlier and proceed to next tj.
#
# 4. For each unmarked tuple ti in the first array, report ti as an outlier.
# 5. If the second array has served as the first array anytime before, stop;
#    Otherwise, swap the names of the first and second arrays and goto step 2.
from pandas import DataFrame


def NL(T: DataFrame, B: float, M: int, D: float, dist):
    pass
    # while True:
    #     T.loc[:, 'outlier'] = None
    #     first = T.sample(frac=B)  # 1.
    #     second = T[T.isin(first) == False]
    #     for i in range(first.shape[0]):  # 2.
    #         count_i = 0
    #         t_i = first.iloc[i, :].squeeze()
    #         for j in range(second.shape[0]):
    #             t_j = second.iloc[j, :].squeeze()
    #             if dist(t_i, t_j) <= D:
    #                 count_i = count_i + 1
    #             if count_i > M:
    #                 T.iloc[i, -1] = False
    #     while not first.empty:  # 3.
