import unittest
import pandas as pd
import numpy as np

from create_partition import random_split
from ModelFactory import GlobalOpts
from create_partition import *

class TestCreatePartition(unittest.TestCase):

    def setUp(self):
        self.opts = GlobalOpts('partition_test')

    def test_random_split(self):
        df = pd.DataFrame({'x':[1,2,3,4,5,6,7,8,9,10]})
        dfA, dfB = random_split(df, proportion=0.6)
        assert dfA.shape[0] == 6
        assert dfB.shape[0] == 4
        recombined = dfA.append(dfB)
        assert (df['x'] == np.sort(recombined['x'].values)).all()

    def test_import_task1_data(self):
        # Check that the columns are correct
        data = import_task1_data(self.opts)
        expected_columns = [
                'report_id',
                'mattlungrenMD',
                'mlungrendc76878f480f48f4',
                'ndm29',
                'pefinder'
                ]
        intersect = data.columns.intersection(expected_columns)
        assert len(intersect) == len(expected_columns), intersect

    def test_import_task2_data(self):
        # Check that the columns are correct

        data = import_task2_data(self.opts)
        expected_columns = [
                'report_id',
                'LOOKING_FOR_PE_label',
                'PE_BURDEN_label'
                ]
        intersect = data.columns.intersection(expected_columns) 
        assert len(intersect) == len(expected_columns), intersect

if __name__ == '__main__':
    unittest.main()
