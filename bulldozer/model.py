#! /usr/bin/env python
"""
Diabetus
"""
from os import path
from dateutil.parser import parse
import pandas


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
        self.feature_coder()
        self.remove_string_features()
        self.open_hdfs()
        self.write_validation()
        self.write_train()

    def data_path(self, fname):
        return path.join(self.data_dir, fname)

    def load_file_handles(self):
        self.train = pandas.read_csv(self.data_path('Train.csv'), index_col='SalesID',
                                     converters={'saledate': parse}
                                     )
        self.valid = pandas.read_csv(self.data_path('Valid.csv'), index_col='SalesID',
                                     converters={'saledate': parse}
                                     )
        self.train = self.train.append(self.valid)

    def feature_coder(self, sparse_keep=50):
        dummy_cols = self.train.columns
        dummy_cols = dummy_cols.drop('SalePrice')
        dummy_cols = dummy_cols.drop('saledate')
        for i in dummy_cols:
            if self.train[i].nunique() > sparse_keep:
                un = self.train.ix[self.valid.index].groupby(i).size()
                un.sort()
                dummy_df = pandas.get_dummies(self.train[i][self.train[i].isin(un[-sparse_keep:].index)])
            else:
                dummy_df = pandas.get_dummies(self.train[i])
            for j in dummy_df.columns:
                self.train[str(i) + str(j)] = dummy_df[j]#.fillna(0)
        self.train['SaleYear'] = [d.year for d in self.train.saledate]
        self.train['SaleMonth'] = [d.month for d in self.train.saledate]
        self.train['SaleDay'] = [d.day for d in self.train.saledate]
        del self.train['saledate']
        nanless_median = self.train.median(axis=0)
        nanless_median = nanless_median.to_dict()
        del nanless_median['SalePrice']
        self.train = self.train.fillna(value=nanless_median)

    def remove_string_features(self):
        self.train = self.train[self.train.dtypes[self.train.dtypes != 'object'].index]

    def open_hdfs(self):
        self.datamart = pandas.HDFStore('dm.hdfs')

    def write_validation(self):
        self.valid = {}
        valid_frame = self.train[self.train[self.target_var].isnull()]
        del valid_frame[self.target_var]
        valid_frame = valid_frame.fillna(0)
        self.valid['index'] = valid_frame.index.tolist()
        self.valid['data'] = valid_frame.as_matrix()
        v = pandas.DataFrame(data=self.valid['data'], index=self.valid['index'])
        self.datamart['valid'] = v

    def write_train(self):
        self.data = {}
        data_frame = self.train[self.train.SalePrice.notnull()]
        target_var = data_frame['SalePrice'].values.copy()
        self.data['target'] = target_var
        del data_frame['SalePrice']
        data_frame = data_frame.fillna(0)
        self.data['data'] = data_frame.as_matrix()
        t = pandas.DataFrame(data=self.data['data'])
        t['target'] = self.data['target']
        self.datamart['data'] = t

def get_train_test_data():
    datamart = pandas.HDFStore('dm.hdfs','r')
    trainset =  datamart['data']
    y = trainset['target'].values
    del trainset['target']
    datamart.close()
    return trainset.as_matrix(), y
