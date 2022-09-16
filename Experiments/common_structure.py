from pandas import DataFrame

from DataPreProcessing.analysis import clean_dataset, transform_dataset, expand_dataset
from Utils.Visualization.visualization import visualize_cluster


class general_experiment:
    """
    This class implements the general behaviour of the conducted experiments.
    """

    def __init__(self, clustering_alg, outlier_dect_alg, dim_reduction_alg, tuning_alg, data):
        self.clustering_alg = clustering_alg
        self.outlier_dect_alg = outlier_dect_alg
        self.dim_reduction_alg = dim_reduction_alg
        self.tuning_alg = tuning_alg
        self.data = data
        self.hyperpar_cand = None

    def transform_data(self) -> None:
        clean_dataset(self.data)
        transform_dataset(self.data)
        pass

    def enhance_data(self) -> None:
        expand_dataset(self.data)
        pass

    def reduce_dimensionality(self, n_final_features: int = 10) -> DataFrame:
        return self.dim_reduction_alg(data=self.data, n_final_features=n_final_features)
        pass

    def tune_cl_alg(self, candidates: [], settings: dict, verbose: int = 0) -> DataFrame:
        hyperpar = self.tuning_alg(A=self.clustering_alg, D=self.data, hyperpar=candidates, settings=settings,
                                   verbose=verbose)
        return self.clustering_alg(self.data, hyperpar, settings)
        pass

    def get_outliers(self, data: DataFrame, settings: dict, kind: str = "LDOF",
                     output_path: str = "Data/Results/Experiments/",
                     i: int = 0) -> None:
        if kind == "LDOF":
            result = self.outlier_dect_alg(data=data, distance=settings['distance'], n=settings['n'],
                                           k=settings['k'])
        elif kind == "CBOD":
            result = self.outlier_dect_alg(data=data, epsilon=settings['epsilon'], k=settings['k'])
        else:
            result = data

        visualize_cluster(data=data, labels=result['cluster'], i=i, h=2, additional="Experiment - Clustering",
                          path=output_path)
        visualize_cluster(data=result, labels=result['cluster'], i=i, h=2, additional="Experiment - Outliers",
                          path=output_path, cluster_or_outliers="outlier")
        pass

    def experiment(self, dimensions: [], candidates: [], settings: dict, verbose: int = 0) -> None:
        self.transform_data()  # Data transformation
        self.enhance_data()  # Data enhancement
        datasets = self.data

        for n in dimensions:  # Dimensionality Reduction
            datasets.append(self.reduce_dimensionality(n_final_features=n))

        for i in range(datasets):  # Hyperparameter tuning
            datasets[i] = self.tuning_alg(candidates=candidates, settings=settings, verbose=verbose)

        for i in range(datasets):  # Outlier detection
            for kind in ["LDOF", "CBOD"]:
                self.get_outliers(data=datasets[i], settings=settings, i=i, kind=kind)
        pass
