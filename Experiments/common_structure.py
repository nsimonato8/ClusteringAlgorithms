from DataPreProcessing.Visualization.visualization import visualize_cluster
from DataPreProcessing.analysis import clean_dataset, transform_dataset, expand_dataset


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

    def transform_data(self):
        clean_dataset(self.data)
        transform_dataset(self.data)
        pass

    def enhance_data(self):
        expand_dataset(self.data)
        pass

    def reduce_dimensionality(self, n_final_features: int = 10):
        self.data = self.dim_reduction_alg(data=self.data, n_final_features=n_final_features)
        pass

    def tune_cl_alg(self, candidates: [], settings: dict, verbose: int = 0):
        hyperpar = self.tuning_alg(A=self.clustering_alg, D=self.data, hyperpar=candidates, settings=settings,
                                   verbose=verbose)
        self.data = self.clustering_alg(self.data, hyperpar, settings)
        pass

    def get_outliers(self, settings: dict, kind: str = "LDOF", output_path: str = "Data/Results/Experiments/",
                     i: int = 0):
        if kind == "LDOF":
            result = self.outlier_dect_alg(data=self.data, distance=settings['distance'], n=settings['n'],
                                           k=settings['k'])
        elif kind == "CBOD":
            result = self.outlier_dect_alg(data=self.data, epsilon=settings['epsilon'], k=settings['k'])
        else:
            result = self.data

        visualize_cluster(data=self.data, labels=self.data['cluster'], i=i, h=2, additional="Experiment - Clustering",
                          path=output_path)
        visualize_cluster(data=result, labels=result['cluster'], i=i, h=2, additional="Experiment - Outliers",
                          path=output_path, cluster_or_outliers="outlier")
        pass

    def experiment(self):

        pass
