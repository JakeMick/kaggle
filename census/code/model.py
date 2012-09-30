#! /usr/bin/env python
"""
Census regressor on Weighted Mean L1
"""
from os import path
import pandas
import numpy as np
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
        self.training = pandas.read_csv(self.training_handle)
        self.testing = pandas.read_csv(self.testing_handle)
        self.training_y = self.training[y_header]
        del self.training[y_header]

    def close_file_handles(self):
        """ Herp.
        """
        self.training_handle.close()
        self.testing_handle.close()
def write_prediction(pred):
    p = processing()
    file_handle = open(path.join(p.sub_dir,'prediction.csv'),'w')
    d = data_io()
    _,labels = d.get_heldout_data()
    for prob, lab in zip(pred,labels):
        file_handle.write('"{0}",{1}\n'.format(lab,prob))
    file_handle.close()

def rf_bench():
    clf = RandomForestClassifier(n_estimators=200,oob_score=True, bootstrap=True, n_jobs=8)
