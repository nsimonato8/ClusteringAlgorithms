# ClusteringAlgorithms

A collection of demos of clustering algorithms. For now, each implementation is naive. The data manipulation was
accomplished by using Pandas.

- - -

## DensityBased

This module contains the implementation(s) of the Density Based clustering algorithm.

At the moment, the following algorithms are available:

* DB-Scan;

## Hierarchical

Here are contained the implementation(s) of the Hierarchical clustering algorithms.

What follows is a list of the available algorithms:

*

## KMeansFamily

Here are contained the implementation(s) of the K-Means-like clustering algorithms.

What follows is a list of the available algorithms:

* K-Means
* K-Medoids

- - -

## OutliersDetection

The functions contained in this module can be used for the detection of outliers in a given dataset.

The following algorithms are available:

* NL algorithm, as proposed by Knorr et al. [1998]

## Tuning

The functions contained in this module can be used for the tuning of the hyperparameters of a given clustering
algorithm.

The following methods are available:

* MATR-CV algorithm, as described by Xinjie Fan et al. [2020]

- - -

## Utils

This module contains secondary functions, that are commonly used by all the other modules.

In particular:

* Distances:
    * Euclidean Distance;
    * Cosine Distance.
* Evaluation functions:
    * Silhouette index.
* Information Gain indexes:
    * Entropy.
    * Has_Changed.