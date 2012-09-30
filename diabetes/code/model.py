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
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.cross_validation import KFold
from ml_metrics import log_loss
from scipy.optimize import fmin_bfgs
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA

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
        self.cores = 8

    def strip_bad_characters(self,string):
        no_unicode = ''.join((c for c in string if 0 < ord(c) < 127))
        no_bad = re.sub('\&.*\;', '', no_unicode)
        return no_bad

    def load_file_handles(self):
        """ Creates connection to database and a helper cursor.
        """
        self.conn = sqlite3.Connection(path.join(self.data_dir,'compData.db'))
        self.curr = sqlite3.Cursor(self.conn)

    def get_tables_column(self, table, column_name):
        super_query = self.conn.execute("SELECT {0} from {1}".format(
            column_name, table))
        return super_query.fetchall()

    def get_tables_patient_column(self, table, column_name):
        super_query = self.conn.execute("SELECT PatientGuid,{0} from {1}".format(
            column_name, table))
        return super_query.fetchall()


    def get_tables_column_where(self, column_name, table, target, stuff):
        super_query = self.conn.execute("SELECT {0} from {1} where {2}={3}".format(
            column_name, table, target, stuff))
        return super_query.fetchall()

    def get_column_names(self, table):
        self.curr.execute("PRAGMA table_info({0})".format(table))
        return self.curr.fetchall()

    def most_unique(self, list_o_stuff, n=500):
        c = Counter(list_o_stuff)
        return [i[0] for i in c.most_common(n=n)]

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

    def split_mcom(self, list_o_stuff):
        """Get the n most common first element split on the periods"""
        split_common_values = self.most_unique(
                [l.split('.')[0] for l in list_o_stuff])
        return split_common_values

    def get_datasets(self):
        tablenames = {
                '{0}_allMeds' : (
                    ('MedicationNdcCode','mcom_0'), ('UserGuid','mcom')
                    ),
                '{0}_allergy' : (
                    ('AllergyType','mcom_0'), ('ReactionName', 'mcom'),
                    ('SeverityName', 'mcom')
                    ),
                '{0}_diagnosis' : (
                    ('ICD9Code','mcom_split_0'), ('Acute','mcom'),
                    ('UserGuid','mcom')
                    ),
                '{0}_labs' : (
                    ('AbnormalFlags','mcom_0'), ('PanelName','mcom'),
                    ('HL7Text', 'mcom')
                    ),
                '{0}_medication' : (
                    ('MedicationNdcCode','mcom_0'),
                        ),
                '{0}_patient' : (
                    ('YearOfBirth', 'raw_0'), ('State', 'mcom'),
                    ('Gender', 'mcom')
                    ),
#                '{0}_patientCondition' : (
#                    ),
#                '{0}_patientSmokingStatus' : (
#                    ),
                '{0}_patientTranscript' : (
                    ('Height', 'raw_0'), ('Weight', 'rignorezero'),
                    ('BMI', 'rignorezero'), ('SystolicBP', 'raw'),
                    ('DiastolicBP', 'raw'), ('RespiratoryRate', 'raw'),
                    ('Temperature', 'raw')
                    ),
                '{0}_prescription' : (
                    ('GenericAllowed','mcom_0'), ('RefillAsNeeded', 'mcom'),
                    ('NumberOfRefills','raw'), ('PrescriptionGuid','mcom')
                    ),
                '{0}_smoke' : (
                    ('SmokingStatus_NISTCode','raw'),
                    ),
#                '{0}_transcript' : (
#                    )
        }
        self.train_patient_info = {}
        self.test_patient_info = {}
        for table, column_nfo in tablenames.items():
            train_table = table.format('training')
            test_table = table.format('test')
            for target_column, what_to_do in column_nfo:
                tr_patient = []
                tr_info = []
                te_patient = []
                te_info = []
                train_patient_info = self.get_tables_column_patient(
                        train_table, target_column)
                test_patient_info = self.get_tables_column_patient(
                        test_table, target_column)
                for p,i in train_patient_info:
                    tr_patient.append(p)
                    tr_info.append(i)
                    self.add_subdict(self.train_patient_info,p)
                for p,i in test_patient_info:
                    te_patient.append(p)
                    te_info.append(i)
                    self.add_subdict(self.test_patient_info,p)
                if what_to_do[:4] == 'mcom':
                    common_elements = self.most_unique(tr_info)
                    for i in common_elements:
                        table_name = '{0}_{1}'.format(table.format(target_column), i)
                        for patient_guid, info in zip(tr_patient,tr_info):
                            if table_name not in self.train_patient_info[patient_guid]:
                                self.train_patient_info[patient_guid][table_name] = 0
                            self.train_patient_info[patient_guid][table_name] += 1
                        for patient_guid, info in zip(te_patient,te_info):
                            if table_name not in self.test_patient_info[patient_guid]:
                                self.test_patient_info[patient_guid][table_name] = 0
                            self.test_patient_info[patient_guid][table_name] += 1

                if what_to_do[5:10] == 'split':
                    common_elements = self.split_mcom(tr_info)
                    for i in common_elements:
                        table_name = '{0}_{1}'.format(table.format(target_column), i)
                        for patient_guid, info in zip(tr_patient,tr_info):
                            if table_name not in self.train_patient_info[patient_guid]:
                                self.train_patient_info[patient_guid][table_name] = 0
                            self.train_patient_info[patient_guid][table_name] += 1
                        for patient_guid, info in zip(te_patient,te_info):
                            if table_name not in self.test_patient_info[patient_guid]:
                                self.test_patient_info[patient_guid][table_name] = 0
                            self.test_patient_info[patient_guid][table_name] += 1

                if what_to_do[-1] == '0':
                    count_of_table = table.format('count')
                    for patient_guid, info in zip(tr_patient,tr_info):
                        if count_of_table not in self.train_patient_info[patient_guid]:
                            self.train_patient_info[patient_guid][count_of_table] = 0
                        self.train_patient_info[patient_guid][count_of_table] += 1
                    for patient_guid, info in zip(te_patient,te_info):
                        if count_of_table not in self.test_patient_info[patient_guid]:
                            self.test_patient_info[patient_guid][count_of_table] = 0
                        self.test_patient_info[patient_guid][count_of_table] += 1
                # FIXME
                if what_to_do[:3] == 'raw':
                    table_name = '{0}_{1}'.format(table.format(target_column), 'raw')
                    for patient_guid, info in zip(tr_patient,tr_info):
                        try:
                            self.train_patient_info[patient_guid][table_name] = float(info)
                        except:
                            self.train_patient_info[patient_guid][table_name] = np.nan
                    for patient_guid, info in zip(te_patient,te_info):
                         try:
                            self.test_patient_info[patient_guid][table_name] = float(info)
                         except:
                            self.test_patient_info[patient_guid][table_name] = np.nan

                if what_to_do[:3] == 'rignorezero':
                    table_name = '{0}_{1}'.format(table.format(target_column), 'raw')
                    for patient_guid, info in zip(tr_patient,tr_info):
                        try:
                            if info == 0:
                                self.train_patient_info[patient_guid][table_name] = np.nan
                            else:
                                self.train_patient_info[patient_guid][table_name] = float(info)
                        except:
                            self.train_patient_info[patient_guid][table_name] = np.nan
                    for patient_guid, info in zip(te_patient,te_info):
                         try:
                            if info == 0:
                                self.test_patient_info[patient_guid][table_name] = np.nan
                            else:
                                self.test_patient_info[patient_guid][table_name] = float(info)
                         except:
                            self.test_patient_info[patient_guid][table_name] = np.nan

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
        self.test_df = pandas.DataFrame(self.test_patient_info)

        del self.train_patient_info
        del self.test_patient_info

        self.train_df = self.train_df.transpose()
        self.train_df = self.train_df.astype(float)

        self.test_df = self.test_df.transpose()
        self.test_df = self.test_df.astype(float)


        self.datamart['train'] = self.train_df
        self.datamart['test'] = self.test_df
        del self.test_df
        del self.train_df

    def get_munged_clean_data(self):
        train_df = self.datamart['train']

        Y_train_df = train_df['Target']
        Y_train = np.array(Y_train_df)
        del train_df['Target']

        test_df = self.datamart['test']

        assert np.all(train_df.columns == test_df.columns)
        X_train = np.array(train_df)
        del train_df
        X_test = np.array(test_df)
        del test_df
        X_train_nan = np.isnan(X_train)
        X_test_nan = np.isnan(X_test)
        X_train = np.hstack((X_train,X_train_nan))
        X_test = np.hstack((X_test,X_test_nan))
        X_train_median = stats.nanmedian(X_train,axis=0)
        for i in xrange(X_train.shape[1]):
            X_train[np.isnan(X_train[:,i]),i] = X_train_median[i]
        for i in xrange(X_test.shape[1]):
            X_test[np.isnan(X_test[:,i]),i] = X_train_median[i]
        keep_not_boring = X_train.std(axis=0) > 0.0
        X_train = X_train[:,keep_not_boring]
        X_test = X_test[:,keep_not_boring]
        return X_train, Y_train, X_test

    def make_processed_data(self):
        X_train, Y_train, X_test = self.get_munged_clean_data()
        print("Dropping singular components from {0} components.".format(
            X_train.shape[1]))
        #p = PCA(whiten=True, n_components=1000)
        #p.fit(X_train)
        #X_train = p.transform(X_train)
        #print("Retained components {0}".format(X_train.shape[1]))
        #X_test = p.transform(X_test)
        feat_clf = ExtraTreesClassifier(n_estimators=1000, bootstrap=True,
                compute_importances=True, oob_score=True, n_jobs=4,
                random_state=21, verbose=1)
        feat_clf.fit(X_train, Y_train)
        feat_path = path.join(
                path.join(self.data_dir,'models'),'feature_selection')
        np.save(path.join(feat_path,'xtrain'),X_train)
        np.save(path.join(feat_path,'ytrain'),Y_train)
        np.save(path.join(feat_path,'xtest'),X_test)
        np.save(path.join(feat_path,'feat_imp'),feat_clf.feature_importances_)

    def write_heldout_indices(self):
        test = self.datamart['test']
        ha = open(path.join(path.join(path.join(self.data_dir,'models'),
            'feature_selection'),'held_index.csv'),'w')
        writeme = csv.writer(ha)
        for row in test.index.tolist():
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
        if n > self.sort_feat_imp.shape[0]:
            return self.feat_imp > 0.0
        else:
            return self.feat_imp > self.sort_feat_imp[n - 1]

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

    def get_train_valid_test(self):
        x,y = self.get_training_data(n=113)
        inds = np.arange(y.shape[0])
        np.random.shuffle(inds)
        X_train, Y_train = x[inds[:8000],:], y[inds[:8000]]
        X_test, Y_test = x[inds[8000:9000],:], y[inds[8000:9000]]
        X_valid, Y_valid = x[inds[9000:],:], y[inds[9000:]]
        all_tup = [(X_train, Y_train), (X_test, Y_test), (X_valid, Y_valid)]
        for i in all_tup:
            for j in i:
                print(j.shape)
        return all_tup

def fit_platt_logreg(score, y):
    """
    https://github.com/paolo-losi/scikit-learn/blob/calibration/scikits/learn/calibration/platt.py
    """
    y = np.asanyarray(y).ravel()
    score = np.asanyarray(score, dtype=np.float64).ravel()

    uniq = np.sort(np.unique(y))

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

    a, b = fmin_bfgs(logloss, [0, 0], ddx_logloss)
    return a / score_std, b - a * score_mean / score_std

def scale_platt_values(score, a, b):
    return 1. / (1. + np.exp(-(a * score + b)))

class PlattScaler(BaseEstimator, ClassifierMixin):
    """Predicting Good Probabilities With Supervised Learning
    https://github.com/paolo-losi/scikit-learn/blob/calibration/scikits/learn/calibration/platt.py
    """

    def __init__(self, classifier):
        self.classifier = classifier
        self.a = None
        self.b = None

    def fit(self, X, y, cv=None, **fit_params):
        self._set_params(**fit_params)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices,:]
        y = y[indices]
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
        print("Optimistic Log-loss: {0:f}".format(
            log_loss(yy, (1./1. + np.exp(-(self.a *scores + self.b))))))
        self.classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        score = self.classifier.predict_proba(X)[:,1].reshape(-1,1)
        proba = 1. / (1. + np.exp(-(self.a * score + self.b)))
        return np.hstack((1. - proba, proba))

    def predict(self, X):
        return self.predict_proba(X) > .5

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

def from_the_top():
    p = processing()
    print("Run")
    p.run()
    print("Process")
    p.make_processed_data()
    p.write_heldout_indices()
