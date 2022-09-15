# ClusteringAlgorithms

This repository contains all the Python modules used for the research work of Niccolò Simonato's bachelor degree.

- - -

## Clustering

### DensityBased

This module contains the implementation(s) of the Density Based clustering algorithm.

At the moment, the following algorithms are available:

* DB-Scan;

### Hierarchical

This module contains the implementation(s) of the Hierarchical clustering algorithm.

At the moment, the following algorithms are available:

* HAC;

### KMeansFamily

Here are contained the implementation(s) of the K-Means-like clustering algorithms.

What follows is a list of the available algorithms:

* K-Means
* K-Medoids

- - -

## DataPreProcessing

This module contains the scripts used for the Data Preprocessing phase of this project.
- - -

## OutliersDetection

The functions contained in this module can be used for the detection of outliers in a given dataset.

The following algorithms are available:

* CBOD algorithm, as proposed by Sheng-yi Jiang et al. [2008];
* LDOF algorithm, as proposed by Abir Smiti [2020];

- - -

## Tuning

The functions contained in this module can be used for the tuning of the hyperparameters of a given clustering
algorithm.

The following methods are available:

* MATR algorithm, as described by Xinjie Fan et al. [2020]
* A simple implementation of the Elbow Method.

- - -

## Utils

This module contains secondary functions, that are commonly used by all the other modules.

In particular:

### Distances

* Euclidean Distance;
* Cosine Distance;
* Minkowski Distance.

### Algebric Operations

* Similarity matrix;
* Clustering matrix;
* Trace of a given matrix.

### Clustering Operations

* Cluster distance for the CBOD algorithm;
* Diff function for the CBOD algorithm;
* Frequency of a value in a given dataset's feature.

- - -

## Experiments

This module implements the experiments conducted in order to analyze the identified outliers.

The following sections will briefly describe how the tests are implemented.

### DBSCAN-based experiments

### HAC-based experiments

### KMeans/KMedoids-based experiments

