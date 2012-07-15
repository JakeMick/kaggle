import numpy as np

from scipy.optimize import fmin_bfgs

from ..base import BaseEstimator, ClassifierMixin
from ..cross_val import KFold


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
        #FIXME check number of classes
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
            score = clf.decision_function(X_test)
            score_list.append(score)
            y_list.append(y_test)

        yy = np.concatenate(y_list)
        scores = np.concatenate(score_list)

        self.a, self.b = fit_platt_logreg(scores, yy)
        self.classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        score = self.classifier.decision_function(X)
        proba = 1. / (1. + np.exp(-(self.a * score + self.b)))
        return np.hstack((1. - proba, proba))

    def predict(self, X):
        #FIXME
        return self.predict_proba(X) > .5
