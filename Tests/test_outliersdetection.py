import sys
import unittest

import pandas as pd
from scipy.spatial.distance import euclidean

from Clustering.KMeansFamily.kmeansfamily import kmeans
from OutliersDetection.CBOD import CBOD
from OutliersDetection.LDOF import top_n_LDOF


class TestOutliersDetection(unittest.TestCase):

    def test_LDOF(self):
        test_data = pd.read_csv("../Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                                engine='python')
        result = top_n_LDOF(data=test_data, distance=euclidean, n=10, k=15)

        # Saving the reference of the standard output
        original_stdout = sys.stdout
        with open(f'[test_LDOF]log_{1}.txt', 'w') as f:
            sys.stdout = f
            print(result.info())
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(
                    f"Number of outliers detected: {result.shape[0]}\tTotal number of instances: {test_data.shape[0]}\nResults:\n{result.head(n=result.shape[0])}")
            # Reset the standard output
            sys.stdout = original_stdout
        pass

    def test_CBOD_KMeans(self):
        test_data = pd.read_csv("../Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                                engine='python')
        settings = {'n_init': 10,
                    'max_iter': 500,
                    'verbose': 0,
                    'algorithm': 'lloyd'}
        param = {'n_clusters': 5}
        result = kmeans(data=test_data, hyperpar=param, settings=settings)

        detection_results = CBOD(data=result, k=param['n_clusters'], epsilon=0.05)

        # Saving the reference of the standard output
        original_stdout = sys.stdout
        with open(f'[test_CBOD-KMeans]log_{2}.txt', 'w') as f:
            sys.stdout = f
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                for i in detection_results:
                    print(f"\t[cluster {i[0]}] Outlier Factor: {i[1]}")
                print(
                    f"Number of outliers detected: {result.loc[result['outlier'] == True, :].shape[0]}\tTotal number of instances: {test_data.shape[0]}\nResults:\n{result.loc[result['outlier'] == True, :].head(n=result.shape[0])}")
            # Reset the standard output
            sys.stdout = original_stdout
        pass


if __name__ == '__main__':
    unittest.main()
