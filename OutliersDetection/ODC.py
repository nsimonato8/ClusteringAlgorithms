"""
The ODC algorithm detects outliers by using a modified version of the well-known K-means.
As a first step, each object is appointed to the closest centroid.
After that the Sum of Squared Error SSE and Total Sum of Squares SST are calculated in order to reduce error and
improve clustering quality. To identify outliers ODC calculates the distance between k centroids and all objects
and then compares it with the mean of k centroids and all object.
If its bigger, the object may be considered as outlier and will be removed.
"""

# Input: Data set D(A1, A2, ...., An), number of clusters k, threshold p.

# 1. Choose a value of k.
# 2. Select k objects randomly and use them as initial set of centroids.
# repeat
# 3. Calculate the distances between k centroids and all the objects in data set D.
# 4. Calculate the mean distances (Md) between k centroids and all the objects in data set D.
# 5. Assign each object to the cluster for which it is nearest centroid and calculate SSE/SST.
# for each object x ∈ data set D do
#     if distance (x, ck) ≻ p ∗ (Md) then
#         6. Consider x as an outlier and remove from data set D and calculate SSE/SST.
#     end if
# end for
# 7. Recalculate the centroids.
# until objects stop changing clusters

# Output: Clustered Data, Outliers and SSE \SST.
