"""
Implementation of the CBOD algorithm, as described by Sheng-yi Jiang et al. [2008]
"""

# Stage 1. Clustering: Cluster on data set D and produce clustering results C ={C1,C2 ,",Ck} .
# Stage 2. Determining Outlier Clusters:
#     Compute outlier factor OF(C_i)(1 <= i <= k) , sort clustersC = {C1,C2 ,..,Ck} according to their OF.
#     Search the minimum b , which satisfies sum(from=0, to=b, |C_i|)/|D| >= epsilon, with 0< = epsilon <= 1
#     Label clusters C_1, C_2, ..., C_b  with ‘outlier’
#     Label clusters C_b+1, C_b+2, ..., C_k  with ‘normal’
