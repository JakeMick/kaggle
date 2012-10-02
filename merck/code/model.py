#! /usr/bin/env python
"""
model.py
Merck Molecular Activity Challenge
"""
from os import path, listdir
import numpy as np
from scipy import stats
import pandas
from sklearn import ensemble

class processing():
    """ Loops over the files and appends predictions.

    Assumes project structure is:
        merck/
            code/
                model.py
            data/
                training/
                    ACT1_competition_training.csv
                    ...
                    ACT15_competition_training.csv
                testing/
                    ACT1_competition_test.csv
                    ...
                    ACT15_competition_test.csv
            submissions/

    Example:
        >>> p = processing(prediction_fname='glm.csv')
        >>> for train_x, train_y, test_x, test_labels in p:
                model.fit(train_x, train_y)
                prediction = model.predict(test_x)
                p.append_prediction(prediction, test_labels)
    """
    def __init__(self, prediction_fname='prediction.csv'):
        """ Defines locations of project management, iteration params.
        """
        current_path = path.dirname(path.abspath(__file__))
        parent_dir = path.dirname(current_path)
        self.prediction_fname = prediction_fname
        self.data_dir = path.join(parent_dir, 'data')
        self.sub_dir = path.join(parent_dir, 'submissions')
        self.out_path = path.join(self.sub_dir, self.prediction_fname)
        self.fail_if_pred_fname_exists()
        self.get_fnames()
        self.start = 0
        self.stop = len(self.training_fnames) - 1

    def __iter__(self):
        return self

    def next(self):
        if self.start > self.stop:
            StopIteration
        elif self.start >= len(self.training_fnames):
            StopIteration
        else:
            cur = self.start
            self.start += 1
            train_x, train_y = self.get_train(cur)
            test_x, test_labels = self.get_test(cur)
            return train_x, train_y,\
                    test_x, test_labels

    def get_fnames(self):
        """ Makes list objects of training and test filenames.
        """
        #  Takes string and returns a integer of the numbers in the string.
        fnk = lambda a: int(''.join([i for i in a if i.isdigit()]))
        self.training_fnames = listdir(path.join(self.data_dir,'training'))
        self.training_fnames.sort(key=fnk)
        self.testing_fnames = listdir(path.join(self.data_dir,'testing'))
        self.testing_fnames.sort(key=fnk)

    def get_train(self,numero):
        train_path = path.join(self.data_dir,'training')
        train_df = pandas.read_csv(
                path.join(train_path,self.training_fnames[numero]))
        print("Current file: %s" % self.training_fnames[numero])
        del train_df['MOLECULE']
        train_y = np.array(train_df['Act'].tolist())
        del train_df['Act']
        train_x = train_df.as_matrix()
        return train_x, train_y

    def get_test(self,numero):
        test_path = path.join(self.data_dir,'testing')
        test_df = pandas.read_csv(
                path.join(test_path,self.testing_fnames[numero]))
        test_labels = test_df['MOLECULE'].tolist()
        del test_df['MOLECULE']
        test_x = test_df.as_matrix()
        return test_x, test_labels

    def fail_if_pred_fname_exists(self):
        if path.isfile(self.prediction_fname):
            raise Exception(
                "The output filename passed to processing already exists.")
        else:
            out_handle = open(self.out_path,'w')
            out_handle.write('"MOLECULE","Prediction"\n')
            out_handle.close()

    def append_prediction(self, prediction, test_labels):
        out_handle = open(self.out_path,'a')
        for label, pred in zip(test_labels, prediction):
            out_handle.write('"%s",' % label)
            out_handle.write('%3.10f\n' % pred)
        out_handle.close()

def r_squared(pred, obs):
    _, _, r, _, _= stats.linregress(pred, obs)
    return r**2

def etr_model():
    """etr.csv
    Leaderboard: 0.41119
    Personal: 0.661851
    """
    print("Processing")
    p = processing(prediction_fname='etr.csv')
    etr = ensemble.ExtraTreesRegressor(bootstrap=True,
            compute_importances=True, oob_score=True,
            n_jobs=4, n_estimators=40)
    all_r = 0
    for train_x, train_y, test_x, test_labels in p:
        print("Fitting model")
        etr.fit(train_x, train_y)
        running_r = r_squared(etr.oob_prediction_, train_y)
        all_r += running_r
        print("R^2 is:" + str(running_r))
        print("Predicting on the test data")
        prediction = etr.predict(test_x)
        print("Writing out the prediction")
        p.append_prediction(prediction, test_labels)
    print("Average R^2:" + str(all_r/15.0))

def many_etr_model():
    """many_etr.csv
    Leaderboard: ???
    Personal: ???
    """
    print("Processing")
    p = processing(prediction_fname='many_etr.csv')
    etr = ensemble.ExtraTreesRegressor(bootstrap=True,
            oob_score=True, n_jobs=4, n_estimators=40)
    rtr = ensemble.RandomForestRegressor(bootstrap=True,
            oob_score=True, n_jobs=4, n_estimators=40)
    all_r = 0
    for train_x, train_y, test_x, test_labels in p:
        print("Fitting model")
        X_ensemble_of_ensembles = []
        X_test_ensemble_of_ensemble = []
        for d in xrange(40):
            print(d)
            etr.fit(train_x, train_y)
            X_ensemble_of_ensembles.append(etr.oob_prediction_)
            X_test_ensemble_of_ensemble.append(etr.predict(test_x))
            rtr.fit(train_x, train_y)
            X_ensemble_of_ensembles.append(rtr.oob_prediction_)
            X_test_ensemble_of_ensemble.append(rtr.predict(test_x))
        X_ensemble_of_ensembles = np.array(X_ensemble_of_ensembles)
        X_test_ensemble_of_ensemble = np.array(X_test_ensemble_of_ensemble)
        big_etr = ensemble.RandomForestRegressor(bootstrap=True,
                oob_score=True, n_jobs=4, n_estimators=200)
        big_etr.fit(X_ensemble_of_ensembles, train_y)
        running_r = r_squared(big_etr.oob_prediction_, train_y)
        all_r += running_r
        print("R^2 is:" + str(running_r))
        print("Predicting on the test data")
        prediction = big_etr.predict(test_x)
        print("Writing out the prediction")
        p.append_prediction(prediction, test_labels)
    print("Average R^2:" + str(all_r/15.0))


if __name__ == "__main__":
    many_etr_model()