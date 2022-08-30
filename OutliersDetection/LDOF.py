"""
The algorithm computes LDOF factor on a data instance which indicate how much it deviates from its neighborhood.
Data instances obtaining high scores are more likely considered as outliers.
LDOF factor is calculated by dividing the KNN distance of an object xp by the KNN inner distance of an object xp.
This file presents an implementation of the LDOF algorithm, as described by Abir Smiti[2020]
"""

# Input: given data set D, natural numbers n and k.

# 1. For each object p in D, retrieve k - nearest neighbors;
# 2. Compute the outlier factor of each object p, The object with LDOF ≺ LODFlb are directly discarded;
# 3. Rank the objects according to their LDOF scores;

# Output: top-n objects with highest LDOF scores.
