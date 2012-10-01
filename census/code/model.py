#! /usr/bin/env python
"""
Census regressor on Weighted Mean L1
"""
import re
from os import path
import pandas
import numpy as np
from scipy import stats
from sklearn import ensemble
from sklearn.cross_validation import KFold

class processing():
    """ Helper class pulls data into numpy array.
    """
    def __init__(self):
        """ Defines general locations of project management.
        """
        current_path = path.dirname(path.abspath(__file__))
        parent_dir = path.dirname(current_path)
        self.data_dir = path.join(parent_dir, 'data')
        self.sub_dir = path.join(parent_dir, 'submissions')
        self.load_file_handles()
        self.load_datesets()
        self.close_file_handles()
        self.remove_junk()
        self.remove_weird_chars()
        #self.convert_to_np()
        self.convert_to_float()
        self.append_clean_nans()

    def load_file_handles(self):
        """ Gets dem files.
        """
        self.training_handle = open(path.join(self.data_dir,
            'training_filev1.csv'),'r')
        self.testing_handle = open(path.join(self.data_dir,
            'test_filev1.csv'),'r')

    def load_datesets(self):
        """ Takes the defined file handles and pulls out the
        data into CSV format.
        """
        y_header = 'Mail_Return_Rate_CEN_2010'
        self.training_x = pandas.read_csv(self.training_handle)
        self.testing_x = pandas.read_csv(self.testing_handle)
        self.training_y = self.training_x[y_header]
        del self.training_x[y_header]

    def close_file_handles(self):
        """ Herp.
        """
        self.training_handle.close()
        self.testing_handle.close()

    def remove_junk(self):
        no_beuno = ['State_name','County_name', 'Flag']
        self.datasets = [self.training_x, self.testing_x]
        for i in no_beuno:
            for hm in self.datasets:
                del hm[i]
        self.train_weights = self.training_x['weight']
        self.test_weights = self.testing_x['weight']


    def convert_to_np(self):
        self.training_x = self.training_x.as_matrix()
        self.testing_x = self.testing_x.as_matrix()
        self.training_y = np.array(self.training_y.tolist())

    def remove_weird_chars(self):
        bad_cols = [i for i in self.training_x.columns if self.training_x[i].dtype == object]
        for i in bad_cols:
            self.training_x[i] = self.training_x[i].str.replace('\$|,','')
            self.testing_x[i] = self.testing_x[i].str.replace('\$|,','')

    def convert_to_float(self):
        self.training_x = self.training_x.astype(float)
        self.testing_x = self.testing_x.astype(float)

    def append_clean_nans(self):
        train_nan = np.isnan(self.training_x)
        train_median = stats.nanmedian(self.training_x)
        train_nan_locs = np.where(train_nan)
        ms, ns = train_nan_locs
        for m, n in zip(ms, ns):
            self.training_x.ix[m,n] = train_median[n]
        cols_to_keep = train_nan.sum(axis=0) != 0
        index_cols_to_keep = cols_to_keep.ix[np.where(cols_to_keep)].index
        self.train_dummy_nan = train_nan[index_cols_to_keep].astype(float)
        n_columns = []
        for i in self.train_dummy_nan.columns.tolist():
            i  = "nan_" + i
            n_columns.append(i)
        self.train_dummy_nan.columns = n_columns
        #self.training_x += self.train_dummy_nan

        test_nan = np.isnan(self.testing_x)
        test_nan_locs = np.where(test_nan)
        ms, ns = test_nan_locs
        for m, n in zip(ms, ns):
            self.testing_x.ix[m,n] = train_median[n]
        self.test_dummy_nan = test_nan[index_cols_to_keep].astype(float)
        self.test_dummy_nan.columns = n_columns
        #self.testing_x += test_dummy_nan

def get_data():
    p = processing()
    etr = ensemble.ExtraTreesRegressor(bootstrap=True,
            compute_importances=True, oob_score=True,
            n_jobs=-1, n_estimators=400)
    etr.fit(p.training_x, p.training_y)
    wmae = (1.0/p.train_weights.sum())*(
            p.train_weights*np.abs(etr.oob_prediction_ - p.training_y)).sum()
    print("WMAE:" + str(wmae) + "\n")
    return etr, p
