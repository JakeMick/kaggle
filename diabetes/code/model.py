#! /usr/bin/env python
"""
Diabetus
"""
from os import path
import sqlite3
import re
from collections import Counter
import csv
import numpy as np
import pandas
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import KFold
from ml_metrics import log_loss
from scipy.optimize import fmin_bfgs
from sklearn.base import BaseEstimator, ClassifierMixin

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
        self.ordinal_threshold = 5
        self.uniq_threshold = 0.001
        self.datamart = pandas.HDFStore(path.join(self.data_dir, 'data.h5'))

    def strip_bad_characters(self,string):
        no_unicode = ''.join((c for c in string if 0 < ord(c) < 127))
        no_html = re.sub('\&.*\;', '', no_unicode)
        return no_html

    def load_file_handles(self):
        """ Creates connection to database and a helper cursor.
        """
        self.conn = sqlite3.Connection(path.join(self.data_dir,'compData.db'))
        self.curr = sqlite3.Cursor(self.conn)

    def get_tables_column(self, table, column_name):
        super_query = self.conn.execute("SELECT {0} from {1}".format(
            column_name, table))

        return super_query.fetchall()

    def get_tables_column_where(self, column_name, table, target, stuff):
        super_query = self.conn.execute("SELECT {0} from {1} where {2}={3}".format(
            column_name, table, target, stuff))
        return super_query.fetchall()

    def get_column_names(self, table):
        self.curr.execute("PRAGMA table_info({0})".format(table))
        return self.curr.fetchall()

    def most_unique(self, list_o_stuff):
        c = Counter(list_o_stuff)
        size = len(list_o_stuff)
        common_stuff = []
        for stuff, count in  c.most_common():
            if float(count)/float(size) <= self.uniq_threshold:
                break
            else:
                common_stuff.append(stuff)
        return common_stuff

    def test_if_fake_int(self, list_o_stuff):
        for v in list_o_stuff:
            value = v[0]
            if type(value) == int or type(value) == float:
                return True
            if value == None:
                continue

    def convert_to_number_list(self, list):
        out = []
        for v in list:
            value = v[0]
            if value == None:
                out.append(np.nan)
            else:
                numba = ''.join([s for s in value if s.isdigit()])
                if numba == '':
                    out.append(np.nan)
                else:
                    out.append(float(numba))

    def is_list_unique(self,list):
        """Determines if the first elements in a list of tuples is unique.
        """
        s = set([])
        for v in list:
            value = v[0]
            if value not in s:
                s.add(value)
            else:
                return False
        return True

    def unique_second_element(self,list):
        """ Returns a list of the unique seconds elements in a list of tuples.
        """
        return set([i[1] for i in list])

    def get_tables_column_patient(self, table, column_name):
        super_query = self.conn.execute("SELECT PatientGuid, {0} FROM {1}".format(
            column_name, table))
        return [s for s in super_query.fetchall() if s[1] != None]

    def get_Y_train(self):
        diabetes_cursor = self.conn.execute("SELECT PatientGuid,\
                dmindicator FROM training_patient")
        diabetes = diabetes_cursor.fetchall()
        return diabetes

    def add_subdict(self, dic, sub_name):
        if sub_name not in dic:
            dic[sub_name] = {}
        return dic

    def get_datasets(self):
        datanames = [
        '{0}_allMeds',
        '{0}_allergy',
        '{0}_diagnosis',
#       '{0}_immunization',
#       '{0}_labObservation',
#       '{0}_labPanel',
        '{0}_labResult',
        '{0}_labs',
        '{0}_medication',
        '{0}_patient',
        '{0}_patientCondition',
        '{0}_patientSmokingStatus',
        '{0}_patientTranscript',
        '{0}_prescription',
        '{0}_smoke',
        '{0}_transcript',
#       '{0}_transcriptAllergy',
#       '{0}_transcriptDiagnosis',
#       '{0}_transcriptMedication'
        ]

        self.train_patient_info = {}
        self.test_patient_info = {}
        for table in datanames:
            train_table = table.format('training')
            test_table = table.format('test')
            columns = self.get_column_names(train_table)

    def run(self):
        self.get_datasets()

        Y_train = self.get_Y_train()
        for patientID, dmindicator in Y_train:
            if patientID not in self.train_patient_info:
                print("{0} ignored".format(patientID))
                pass
            else:
                self.train_patient_info[patientID]['Target'] = dmindicator

        self.train_df = pandas.DataFrame(self.train_patient_info)
        del self.train_patient_info
        self.test_df = pandas.DataFrame(self.test_patient_info)
        del self.test_patient_info

        del self.train_df['patientGuid']
        indices = self.train_df.index.tolist()
        self.train_df = self.train_df.transpose()
        for i in indices:
            self.train_df[i][self.train_df[i] == 'NULL'] = np.nan
        self.train_df = self.train_df.astype(float)

        easy_rm = []
        for i in indices:
            if re.search('PatientGuid', i):
                easy_rm.append(i)
        for i in easy_rm:
            del self.train_df[i]

        easy_rm = []
        for i in indices:
            if re.search('DUMMY', i):
                easy_rm.append(i)

        index_info = set([])
        for i in easy_rm:
            lol = i.split('_')[0]
            if lol not in index_info:
                index_info.add(lol)
        index_dict = {}
        for i in index_info:
            index_dict[i] = []
            for j in easy_rm:
                if re.match(i, j.split('_')[0]):
                    index_dict[i].append(j)

        for name,sub in index_dict.items():
            self.train_df[name+'_max'] = self.train_df[sub].max(axis=1, skipna=True)
            self.train_df[name+'_min'] = self.train_df[sub].min(axis=1, skipna=True)
            self.train_df[name+'_median'] = self.train_df[sub].median(axis=1, skipna=True)
            self.train_df[name+'_std'] = self.train_df[sub].std(axis=1, skipna=True)
            for i in sub:
                del self.train_df[i]

        del self.test_df['patientGuid']
        indices = self.test_df.index.tolist()
        self.test_df = self.test_df.transpose()
        for i in indices:
            self.test_df[i][self.test_df[i] == 'NULL'] = np.nan
        self.test_df = self.test_df.astype(float)

        easy_rm = []
        for i in indices:
            if re.search('PatientGuid', i):
                easy_rm.append(i)
        for i in easy_rm:
            del self.test_df[i]

        easy_rm = []
        for i in indices:
            if re.search('DUMMY', i):
                easy_rm.append(i)

        index_info = set([])
        for i in easy_rm:
            lol = i.split('_')[0]
            if lol not in index_info:
                index_info.add(lol)
        index_dict = {}
        for i in index_info:
            index_dict[i] = []
            for j in easy_rm:
                if re.match(i, j.split('_')[0]):
                    index_dict[i].append(j)

        for name,sub in index_dict.items():
            self.test_df[name+'_max'] = self.test_df[sub].max(axis=1, skipna=True)
            self.test_df[name+'_min'] = self.test_df[sub].min(axis=1, skipna=True)
            self.test_df[name+'_median'] = self.test_df[sub].median(axis=1, skipna=True)
            self.test_df[name+'_std'] = self.test_df[sub].std(axis=1, skipna=True)
            for i in sub:
                del self.test_df[i]
        self.datamart['train'] = self.train_df
        self.datamart['test'] = self.test_df

    def get_munged_clean_data(self):
        train_df = self.datamart['train']

        Y_train_df = train_df['Target']
        Y_train = np.array(Y_train_df)
        del train_df['Target']

        test_df = self.datamart['test']

        assert np.all(train_df.columns == test_df.columns)
        X_train = np.array(train_df)
        X_test = np.array(test_df)
        X_train_nan = np.isnan(X_train)
        X_test_nan = np.isnan(X_test)
        X_train = np.hstack((X_train,X_train_nan))
        X_test = np.hstack((X_test,X_test_nan))
        X_train[np.isnan(X_train)] = 0.0
        X_test[np.isnan(X_test)] = 0.0

        return X_train, Y_train, X_test

    def make_processed_data(self):
        density_keep = 0.000005
        X_train, Y_train, X_test = self.get_munged_clean_data()
        keep = []
        col_length = X_train.shape[1]
        for i in xrange(col_length):
            if (X_train[:,i] != 0).sum() > density_keep:
                keep.append(i)
        X_train = X_train[:,keep]
        X_test = X_test[:,keep]
        feat_clf = ExtraTreesClassifier(n_estimators=1000, bootstrap=True,
                compute_importances=True, oob_score=True, n_jobs=4,
                random_state=21, verbose=1)
        feat_clf.fit(X_train, Y_train)
        feat_path = path.join(path.join(self.data_dir,'models'),'feature_selection')
        np.save(path.join(feat_path,'xtrain'),X_train)
        np.save(path.join(feat_path,'ytain'),Y_train)
        np.save(path.join(feat_path,'xtest'),X_test)
        np.save(path.join(feat_path,'feat_imp'),feat_clf.feature_importances_)

    def write_heldout_indices(self):
        train = self.datamart['test']
        ha = open(path.join(path.join(path.join(self.data_dir,'models'),
            'feature_selection'),'held_index.csv'),'w')
        writeme = csv.writer(ha)
        for row in train.index.tolist():
            writeme.writerow(row)
        ha.close()

class data_io:
    def __init__(self):
        self.model = processing()
        self.feat_path = path.join(path.join(self.model.data_dir,'models'),'feature_selection')
        self.feat_imp = np.load(path.join(self.feat_path,'feat_imp.npy'))
        self.sort_feat_imp = self.feat_imp.copy()
        self.sort_feat_imp.sort()
        self.sort_feat_imp = self.sort_feat_imp[::-1]

    def _n_important_features(self,n=100):
        return self.feat_imp > self.sort_feat_imp[n]

    def get_training_data(self,n=100):
        X_train = np.load(path.join(self.feat_path,'xtrain.npy'))
        Y_train = np.load(path.join(self.feat_path,'ytrain.npy'))
        self.observations = X_train.shape[0]
        return X_train[:,self._n_important_features(n=n)], Y_train

    def get_cv(self, k=9):
        return KFold(n=self.observations, k=k)

    def get_heldout_data(self,n=100):
        label_handle = open(path.join(self.feat_path,'held_index.csv'),'r')
        reader = csv.reader(label_handle)
        labels = []
        for row in reader:
            labels.append(''.join([q for q in row]))
        label_handle.close()

        heldout = np.load(path.join(self.feat_path,'xtest.npy'))
        self.labels = labels
        return heldout[:,self._n_important_features(n=n)], labels


def fit_platt_logreg(score, y):
    y = np.asanyarray(y).ravel()
    score = np.asanyarray(score, dtype=np.float64).ravel()

    uniq = np.sort(np.unique(y))
    if np.size(uniq) != 2:
        raise ValueError('only binary classification is supported. classes: %s'
                                                                        % uniq)

    # the score is standardized to make logloss and ddx_logloss
    # numerically stable
    score_std = score.std()
    score_mean = score.mean()
    score_std = 1
    score_mean = 0
    n_score = (score - score_mean) / score_std

    n = y == uniq[0]
    p = y == uniq[1]
    n_n = float(np.sum(n))
    n_p = float(np.sum(p))
    yy = np.empty(y.shape, dtype=np.float64)
    yy[n] = 1. / (2. + n_n)
    yy[p] = (1. + n_p) / (2. + n_p)
    one_minus_yy = 1 - yy

    def logloss(x):
        a, b = x
        z = a * n_score + b
        ll_p = np.log1p(np.exp(-z))
        ll_n = np.log1p(np.exp(z))
        return (one_minus_yy * ll_n + yy * ll_p).sum()

    def ddx_logloss(x):
        a, b = x
        z = a * n_score + b
        exp_z = np.exp(z)
        exp_m_z = np.exp(-z)
        dda_ll_p = -n_score / (1 + exp_z)
        dda_ll_n = n_score / (1 + exp_m_z)
        ddb_ll_p = -1 / (1 + exp_z)
        ddb_ll_n = 1 / (1 + exp_m_z)
        dda_logloss = (one_minus_yy * dda_ll_n + yy * dda_ll_p).sum()
        ddb_logloss = (one_minus_yy * ddb_ll_n + yy * ddb_ll_p).sum()
        gradient = np.array([dda_logloss, ddb_logloss])
        return gradient

    # FIXME check if fmin_bfgs converges
    a, b = fmin_bfgs(logloss, [0, 0], ddx_logloss)
    return a / score_std, b - a * score_mean / score_std


class PlattScaler(BaseEstimator, ClassifierMixin):
    """Predicting Good Probabilities With Supervised Learning"""

    def __init__(self, classifier):
        self.classifier = classifier
        self.a = None
        self.b = None

    def fit(self, X, y, cv=None, **fit_params):
        self._set_params(**fit_params)
        if cv is None:
            cv = KFold(y.size, k=5)

        clf = self.classifier
        score_list = []
        y_list = []
        for train_index, test_index in cv:
            print train_index.shape, test_index.shape
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf.fit(X_train, y_train)
            score = clf.predict_proba(X_test)[:,1].reshape(-1,1)
            score_list.append(score)
            y_list.append(y_test)

        yy = np.concatenate(y_list)
        scores = np.concatenate(score_list)

        self.a, self.b = fit_platt_logreg(scores, yy)
        self.classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        score = self.classifier.predict_proba(X)[:,1].reshape(-1,1)
        proba = 1. / (1. + np.exp(-(self.a * score + self.b)))
        return np.hstack((1. - proba, proba))

    def predict(self, X):
        #FIXME
        return self.predict_proba(X) > .5

def gradient_boost_model(n=1000):
    dio = data_io()
    x,y = dio.get_training_data(n=n)
    held,labels = dio.get_heldout_data(n=n)
    x_mean = x.mean(axis=0)
    x -= x_mean
    x_std = x.std(axis=0)
    x /= x_std
    held -= x_mean
    held /= x_std
    cv = dio.get_cv()
    sc = PlattScaler(GradientBoostingClassifier(n_estimators=1000,max_depth=2,
        random_state=21,learn_rate=0.1,subsample=0.5))
    losses = []
    heldpred = []
    for train,test in cv:
        X_train, Y_train = x[train],y[train]
        X_test, Y_test = x[test],y[test]
        sc.fit(X_train,Y_train)
        pred = sc.predict_proba(X_test)
        heldpred.append(sc.predict_proba(held)[:,1])
        ll = log_loss(Y_test, pred[:,1])
        print(ll)
        losses.append(ll)
    print("Mean score is {0}".format(np.array(ll).mean()))
    final_predictions = np.vstack(heldpred).mean(axis=0)
    submission_handle = open(path.join(dio.model.sub_dir,'gradient_model1.csv'),'w')
    for lab, pr in zip(labels, final_predictions):
        submission_handle.write('"{0}",{1}'.format(lab,pr))
    submission_handle.close()

