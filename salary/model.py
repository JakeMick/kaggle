#! /usr/bin/env python
"""
Diabetus
"""
from os import path
import pickle
import numpy as np
import pandas
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from scipy.io import mmwrite, mmread
from scipy import sparse
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model import PassiveAggressiveRegressor, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from ml_metrics import mae
import warnings
from itertools import cycle, izip
from sklearn.utils import gen_even_slices
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin
from abc import ABCMeta, abstractmethod
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree.tree import BaseDecisionTree
from sklearn.utils import check_arrays, check_random_state
from sklearn.metrics import accuracy_score, r2_score


class BaseWeightBoosting(BaseEnsemble):
    """Base class for AdaBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=50,
                 estimator_params=tuple(),
                 learning_rate=1.):

        super(BaseWeightBoosting, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.estimator_weights_ = None
        self.estimator_errors_ = None
        self.learning_rate = learning_rate

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
        The training input samples.

        y : array-like of shape = [n_samples]
        The target values (integers that correspond to classes in
        classification, real numbers in regression).

        sample_weight : array-like of shape = [n_samples], optional
        Sample weights. If None, the sample weights are initialized to
        1 / n_samples.

        Returns
        -------
        self : object
        Returns self.
        """
            # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        # Check data
        X, y = check_arrays(X, y, sparse_format="dense")

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float)
            sample_weight[:] = 1. / X.shape[0]
        else:
            # Normalize existing weights
            sample_weight = np.copy(sample_weight) / sample_weight.sum()

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float)

        # Create argsorted X for fast tree induction
        X_argsorted = None

        if isinstance(self.base_estimator, BaseDecisionTree):
            X_argsorted = np.asfortranarray(
                np.argsort(X.T, axis=1).astype(np.int32).T)

        for iboost in xrange(self.n_estimators):
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                X_argsorted=X_argsorted)

            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, X_argsorted=None):
        """Implement a single boost.

    Warning: This method needs to be overriden by subclasses.

    Parameters
    ----------
    iboost : int
    The index of the current boost iteration.

    X : array-like of shape = [n_samples, n_features]
    The training input samples.

    y : array-like of shape = [n_samples]
    The target values (integers that correspond to classes).

    sample_weight : array-like of shape = [n_samples]
    The current sample weights.

    X_argsorted : array-like, shape = [n_samples, n_features] (optional)
    Each column of ``X_argsorted`` holds the row indices of ``X``
    sorted according to the value of the corresponding feature
    in ascending order.
    The argument is supported to enable multiple decision trees
    to share the data structure and to avoid re-computation in
    tree ensembles. For maximum efficiency use dtype np.int32.

    Returns
    -------
    sample_weight : array-like of shape = [n_samples] or None
    The reweighted sample weights.
    If None then boosting has terminated early.

    estimator_weight : float
    The weight for the current boost.
    If None then boosting has terminated early.

    error : float
    The classification error for the current boost.
    If None then boosting has terminated early.
    """
        pass

    def staged_score(self, X, y):
        """Return staged scores for X, y.

    This generator method yields the ensemble score after each iteration of
    boosting and therefore allows monitoring, such as to determine the
    score on a test set after each boost.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
    Training set.

    y : array-like, shape = [n_samples]
    Labels for X.

    Returns
    -------
    z : float
    """
        for y_pred in self.staged_predict(X):
            if isinstance(self, ClassifierMixin):
                yield accuracy_score(y, y_pred)
            else:
                yield r2_score(y, y_pred)

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
        feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        try:
            norm = self.estimator_weights_.sum()
            return (sum(weight * clf.feature_importances_ for weight, clf
                    in zip(self.estimator_weights_, self.estimators_))
                    / norm)

        except AttributeError:
            raise AttributeError(
                "Unable to compute feature importances "
                "since base_estimator does not have a "
                "feature_importances_ attribute")


def _samme_proba(estimator, n_classes, X):
    """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    """
    proba = estimator.predict_proba(X)

    # Displace zero probabilities so the log is defined.
    # Also fix negative elements which may occur with
    # negative sample weights.
    proba[proba <= 0] = 1e-5
    log_proba = np.log(proba)

    return (n_classes - 1) * (log_proba - (1. / n_classes)
                           * log_proba.sum(axis=1)[:, np.newaxis])


class AdaBoostRegressor(BaseWeightBoosting, RegressorMixin):
    """An AdaBoost regressor.

    An AdaBoost [1] regressor is a meta-estimator that begins by fitting a
    regressor on the original dataset and then fits additional copies of the
    regressor on the same dataset but where the weights of instances are
    adjusted according to the error of the current prediction. As such,
    subsequent regressors focus more on difficult cases.

    This class implements the algorithm known as AdaBoost.R2 [2].

    Parameters
    ----------
    base_estimator : object, optional (default=DecisionTreeRegressor)
    The base estimator from which the boosted ensemble is built.
    Support for sample weighting is required.

    n_estimators : integer, optional (default=50)
    The maximum number of estimators at which boosting is terminated.
    In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.)
    Learning rate shrinks the contribution of each regressor by
    ``learning_rate``. There is a trade-off between ``learning_rate`` and
    ``n_estimators``.

    loss : {'linear', 'square', 'exponential'}, optional (default='linear')
    The loss function to use when updating the weights after each
    boosting iteration.

    random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

    Attributes
    ----------
    `estimators_` : list of classifiers
    The collection of fitted sub-estimators.

    `estimator_weights_` : array of floats
    Weights for each estimator in the boosted ensemble.

    `estimator_errors_` : array of floats
    Regression error for each estimator in the boosted ensemble.

    `feature_importances_` : array of shape = [n_features]
    The feature importances if supported by the ``base_estimator``.

    See also
    --------
    AdaBoostClassifier, GradientBoostingRegressor, DecisionTreeRegressor

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
    on-Line Learning and an Application to Boosting", 1995.

    .. [2] H. Drucker, "Improving Regressors using Boosting Techniques", 1997.

    """
    def __init__(self,
                 base_estimator=DecisionTreeRegressor(max_depth=3),
                 n_estimators=50,
                 learning_rate=1.,
                 loss='linear',
                 random_state=None):

        super(AdaBoostRegressor, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate)

        self.loss = loss
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Build a boosted regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
        The training input samples.

        y : array-like of shape = [n_samples]
        The target values (real numbers).

        sample_weight : array-like of shape = [n_samples], optional
        Sample weights. If None, the sample weights are initialized to
        1 / n_samples.

        Returns
        -------
        self : object
        Returns self.
        """
        # Check that the base estimator is a regressor
        if not isinstance(self.base_estimator, RegressorMixin):
            raise TypeError("base_estimator must be a "
                            "subclass of RegressorMixin")

        if self.loss not in ('linear', 'square', 'exponential'):
            raise ValueError(
                "loss must be 'linear', 'square', or 'exponential'")

        # Fit
        return super(AdaBoostRegressor, self).fit(X, y, sample_weight)

    def _boost(self, iboost, X, y, sample_weight, X_argsorted=None):
        """Implement a single boost for regression

        Perform a single boost according to the AdaBoost.R2 algorithm and
        return the updated sample weights.

        Parameters
        ----------
        iboost : int
        The index of the current boost iteration.

        X : array-like of shape = [n_samples, n_features]
        The training input samples.

        y : array-like of shape = [n_samples]
        The target values (integers that correspond to classes in
        classification, real numbers in regression).

        sample_weight : array-like of shape = [n_samples]
        The current sample weights.

        X_argsorted : array-like, shape = [n_samples, n_features] (optional)
        Each column of ``X_argsorted`` holds the row indices of ``X``
        sorted according to the value of the corresponding feature
        in ascending order.
        The argument is supported to enable multiple decision trees
        to share the data structure and to avoid re-computation in
        tree ensembles. For maximum efficiency use dtype np.int32.

        Returns
        -------
        sample_weight : array-like of shape = [n_samples] or None
        The reweighted sample weights.
        If None then boosting has terminated early.

        estimator_weight : float
        The weight for the current boost.
        If None then boosting has terminated early.

        estimator_error : float
        The regression error for the current boost.
        If None then boosting has terminated early.
        """
        estimator = self._make_estimator()

        generator = check_random_state(self.random_state)

        # Weighted sampling of the training set with replacement
        # For NumPy >= 1.7.0 use np.random.choice
        cdf = sample_weight.cumsum()
        cdf /= cdf[-1]
        uniform_samples = generator.random_sample(X.shape[0])
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        # searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        # X_argsorted is not used since bootstrap copies are used.
        estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
        y_predict = estimator.predict(X)

        error_vect = np.abs(y_predict - y)
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect /= error_vect.max()

        if self.loss == 'square':
            error_vect *= error_vect
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        # Calculate the average loss
        estimator_error = (sample_weight * error_vect).sum()

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1., 0.

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        beta = estimator_error / (1. - estimator_error)

        # Boost weight using AdaBoost.R2 alg
        estimator_weight = self.learning_rate * np.log(1. / beta)

        if not iboost == self.n_estimators - 1:
            sample_weight *= np.power(
                beta,
                (1. - error_vect) * self.learning_rate)

        return sample_weight, estimator_weight, estimator_error

    def _get_median_predict(self, X, limit=-1):
        if not self.estimators_:
            raise RuntimeError(
                ("{0} is not initialized. "
                 "Perform a fit first").format(self.__class__.__name__))

        if limit < 1:
            limit = len(self.estimators_)

        # Evaluate predictions of all estimators
        predictions = np.array([
            est.predict(X) for est in self.estimators_[:limit]]).T

        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)

        # Find index of median prediction for each sample
        weight_cdf = self.estimator_weights_[sorted_idx].cumsum(axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)
        median_estimators = sorted_idx[np.arange(len(X)), median_idx]

        # Return median predictions
        return predictions[np.arange(len(X)), median_estimators]

    def predict(self, X):
        """Predict regression value for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
        The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
        The predicted regression values.
        """
        return self._get_median_predict(X)

    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the classifiers in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
        The input samples.

        Returns
        -------
        y : generator of array, shape = [n_samples]
        The predicted regression values.
        """
        for i in xrange(len(self.estimators_)):
            yield self._get_median_predict(X, limit=i + 1)


def _softmax(x):
    np.exp(x, x)
    x /= np.sum(x, axis=1)[:, np.newaxis]


def _tanh(x):
    np.tanh(x, x)


def _dtanh(x):
    """Derivative of tanh as a function of tanh."""
    x *= -x
    x += 1


class BaseMLP(BaseEstimator):
    """Base class for estimators base on multi layer
    perceptrons."""

    def __init__(self, n_hidden, lr, l2decay, loss, output_layer, batch_size,
                 use_dropout=False, dropout_fraction=0.5, verbose=0):
        self.n_hidden = n_hidden
        self.lr = lr
        self.l2decay = l2decay
        self.loss = loss
        self.batch_size = batch_size
        self.use_dropout = use_dropout
        self.dropout_fraction = dropout_fraction
        self.verbose = verbose

        # check compatibility of loss and output layer:
        if output_layer == 'softmax' and loss != 'cross_entropy':
            raise ValueError('Softmax output is only supported ' +
                             'with cross entropy loss function.')
        if output_layer != 'softmax' and loss == 'cross_entropy':
            raise ValueError('Cross-entropy loss is only ' +
                             'supported with softmax output layer.')

        # set output layer and loss function
        if output_layer == 'linear':
            self.output_func = id
        elif output_layer == 'softmax':
            self.output_func = _softmax
        elif output_layer == 'tanh':
            self.output_func = _tanh
        else:
            raise ValueError("'output_layer' must be one of " +
                             "'linear', 'softmax' or 'tanh'.")

        if not loss in ['cross_entropy', 'square', 'crammer_singer']:
            raise ValueError("'loss' must be one of " +
                             "'cross_entropy', 'square' or 'crammer_singer'.")
            self.loss = loss

    def fit(self, X, y, max_epochs=1000, shuffle_data=True, staged_sample=None, verbose=0):
        # get all sizes
        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError("Shapes of X and y don't fit.")
        self.n_outs = y.shape[1]
        # n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        n_batches = n_samples / self.batch_size
        if n_samples % self.batch_size != 0:
            warnings.warn("Discarding some samples: \
                sample size not divisible by chunk size.")
        n_iterations = int(max_epochs * n_batches)

        if shuffle_data:
            X, y = shuffle(X, y)

        # generate batch slices
        batch_slices = list(
            gen_even_slices(n_batches * self.batch_size, n_batches))

        # generate weights.
        # TODO: smart initialization
        self.weights1_ = np.random.uniform(
            size=(n_features, self.n_hidden)) / np.sqrt(n_features)
        self.bias1_ = np.zeros(self.n_hidden)
        self.weights2_ = np.random.uniform(
            size=(self.n_hidden, self.n_outs)) / np.sqrt(self.n_hidden)
        self.bias2_ = np.zeros(self.n_outs)

        # preallocate memory
        x_hidden = np.empty((self.batch_size, self.n_hidden))
        delta_h = np.empty((self.batch_size, self.n_hidden))
        x_output = np.empty((self.batch_size, self.n_outs))
        delta_o = np.empty((self.batch_size, self.n_outs))

        self.oo_score = []
        # main loop
        for i, batch_slice in izip(xrange(n_iterations), cycle(batch_slices)):
            self._forward(i, X, batch_slice, x_hidden, x_output, testing=False)
            self._backward(
                i, X, y, batch_slice, x_hidden, x_output, delta_o, delta_h)
            if staged_sample is not None:
                self.oo_score.append(self.predict(staged_sample))
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        x_hidden = np.empty((n_samples, self.n_hidden))
        x_output = np.empty((n_samples, self.n_outs))
        self._forward(None, X, slice(0, n_samples), x_hidden, x_output,
                      testing=True)
        return x_output

    def _forward(self, i, X, batch_slice, x_hidden, x_output, testing=False):
        """Do a forward pass through the network"""
        if self.use_dropout:
            if testing:
                weights1_ = self.weights1_ * (1 - self.dropout_fraction)
                bias1_ = self.bias1_ * (1 - self.dropout_fraction)
                weights2_ = self.weights2_ * (1 - self.dropout_fraction)
            else:
                dropped = np.random.binomial(1, self.dropout_fraction, self.n_hidden)
                weights1_ = self.weights1_ * dropped
                bias1_ = self.bias1_ * dropped
                weights2_ = (dropped * self.weights2_.T).T
        else:
            weights1_ = self.weights1_
            bias1_ = self.bias1_
            weights2_ = self.weights2_
        x_hidden[:] = np.dot(X[batch_slice], weights1_)
        x_hidden += bias1_
        np.tanh(x_hidden, x_hidden)
        x_output[:] = np.dot(x_hidden, weights2_)
        x_output += self.bias2_

        # apply output nonlinearity (if any)
        self.output_func(x_output)

    def _backward(self, i, X, y, batch_slice, x_hidden, x_output, delta_o, delta_h):
        """Do a backward pass through the network and update the weights"""

        # calculate derivative of output layer
        if self.loss in ['cross_entropy'] or (self.loss == 'square' and self.output_func == id):
            delta_o[:] = y[batch_slice] - x_output
        elif self.loss == 'crammer_singer':
            raise ValueError("Not implemented yet.")
            delta_o[:] = 0
            delta_o[y[batch_slice], np.ogrid[len(batch_slice)]] -= 1
            delta_o[np.argmax(x_output - np.ones((1))[y[batch_slice], np.ogrid[len(batch_slice)]], axis=1), np.ogrid[len(batch_slice)]] += 1

        elif self.loss == 'square' and self.output_func == _tanh:
            delta_o[:] = (y[batch_slice] - x_output) * _dtanh(x_output)
        else:
            raise ValueError(
                "Unknown combination of output function and error.")

        if self.verbose > 0:
            print(np.linalg.norm(delta_o / self.batch_size))
        delta_h[:] = np.dot(delta_o, self.weights2_.T)

        # update weights
        self.weights2_ += self.lr / self.batch_size * np.dot(
            x_hidden.T, delta_o)
        self.bias2_ += self.lr * np.mean(delta_o, axis=0)
        self.weights1_ += self.lr / self.batch_size * np.dot(
            X[batch_slice].T, delta_h)
        self.bias1_ += self.lr * np.mean(delta_h, axis=0)


class MLPRegressor(BaseMLP, RegressorMixin):
    """ Multilayer Perceptron Regressor.

    Uses a neural network with one hidden layer.


    Parameters
    ----------


    Attributes
    ----------

    Notes
    -----


    References
    ----------"""
    def __init__(self, n_hidden=2000, lr=0.1, l2decay=0, loss='square',
                 output_layer='linear', batch_size=2000, use_dropout=True,
                 dropout_fraction=0.5, verbose=0):
        super(MLPRegressor, self).__init__(n_hidden, lr, l2decay, loss,
                                           output_layer, batch_size, use_dropout,
                                           dropout_fraction, verbose)

    def fit(self, X, y, max_epochs=10, shuffle_data=True, staged_sample=None):
        super(MLPRegressor, self).fit(
            X, y, max_epochs,
            shuffle_data, staged_sample)
        return self

    def predict(self, X):
        return super(MLPRegressor, self).predict(X)


class Processing():
    """ Helper class pulls data into numpy array.
    """
    def __init__(self):
        """ Defines general locations of project management.
        """
        current_path = path.dirname(path.abspath(__file__))
        parent_dir = path.dirname(current_path)
        self.data_dir = path.join(parent_dir, 'data')
        self.sub_dir = path.join(parent_dir, 'submissions')
        self.path_to_database = path.join(self.data_dir, 'dm.hdfs')
        self.open_hdfs()

    def make_data(self):
        self.open_hdfs(writeable=True)
        self.configurator()
        self.load_file_handles()
        self.destroy_useless()
        self.dummy_coder()
        self.cheesy_language_model()
        self.drop_non_numerics()
        self.write_validation()
        self.write_train()

    def make_text_data(self):
        self.configurator()
        self.load_file_handles()
        self.write_text_feats()

    def make_char_data(self):
        self.configurator()
        self.load_file_handles()
        self.write_char_feats()

    def configurator(self):
        """ Defines project specific stuff
        """
        self.index_col = 'Id'
        self.dummy_cols = ['Category', 'Company', 'ContractTime', 'ContractType', 'LocationNormalized', 'SourceName']
        self.useless_cols = ['SalaryRaw']
        self.target_var = 'SalaryNormalized'
        self.nlp_cols = {'Title': 100, 'LocationRaw': 100}
        self.raw_text = ['FullDescription', 'Title', 'LocationRaw']

    def open_hdfs(self, writeable=False):
        if writeable:
            can_opener = 'w'
        else:
            can_opener = 'r'
        self.datamart = pandas.HDFStore(self.path_to_database, can_opener)

    def destroy_useless(self):
        for i in self.useless_cols:
            del self.train[i]

    def data_path(self, fname):
        return path.join(self.data_dir, fname)

    def load_file_handles(self):
        self.train = pandas.read_csv(self.data_path('Train_rev1.csv'))
        valid = pandas.read_csv(self.data_path('Valid_rev1.csv'))
        self.train = self.train.append(valid, ignore_index=True)
        self.train.index = self.train[self.index_col]
        del self.train[self.index_col]

    def dummy_coder(self, sparse_keep=50):
        for i in self.dummy_cols:
            un = self.train.groupby(i).size()
            un.sort()
            un = un[-sparse_keep:]
            dummy_df = pandas.get_dummies(self.train[i][self.train[i].isin(un.index)])
            for j in dummy_df.columns:
                self.train[str(i) + str(j)] = dummy_df[j]
            del self.train[i]

    def cheesy_language_model(self):
        for col, spar in self.nlp_cols.items():
            counter = CountVectorizer(max_features=spar, ngram_range=(1, 1))
            count_trans = counter.fit_transform(self.train[col].dropna().values).toarray()
            count_features = pandas.DataFrame(
                index=self.train[col].dropna().index,
                columns=map(lambda i: col + i, counter.get_feature_names()),
                data=count_trans)
            self.train = pandas.concat([self.train, count_features], axis=1, join='outer')

    def drop_non_numerics(self):
        self.train = self.train.drop(self.train.dtypes[(self.train.dtypes == 'object')].index, axis=1)

    def write_validation(self):
        valid_frame = self.train[self.train[self.target_var].isnull()].copy()
        del valid_frame[self.target_var]
        valid_frame = valid_frame.fillna(0)
        self.datamart['valid'] = valid_frame

    def write_train(self):
        data_frame = self.train[self.train[self.target_var].notnull()].copy()
        data_frame = data_frame.fillna(0)
        self.datamart['data'] = data_frame

    def vectorizer(self, col):
        vocab = CountVectorizer(ngram_range=(1, 2), charset_error='replace')
        vocab.fit(self.train[col].fillna('').values)
        return vocab.transform(self.train[col].fillna('').values).tocsr()

    def char_vectorizer(self, col):
        vocab = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 3),
                                  charset_error='replace')
        return vocab.transform(self.train[col].fillna('').values).tocsr()

    def write_char_feats(self):
        t = self.char_vectorizer('FullDescription')
        with open(self.data_path('char_features.mtx'), 'w') as fhand:
            mmwrite(fhand, t)

    def write_hash_feats(self):
        t = self.hash_vectorizer('FullDescription')
        with open(self.data_path('hash_features.mtx'), 'w') as fhand:
            mmwrite(fhand, t)

    def write_text_feats(self):
        all_col = []
        for col in self.raw_text:
            all_col.append(self.vectorizer(col))
            print(all_col[-1].shape)
        t = sparse.hstack(all_col)
        with open(self.data_path('text_features.mtx'), 'w') as fhand:
            mmwrite(fhand, t)

    def get_char_data(self):
        with open(self.data_path('char_features.mtx'), 'r') as fhand:
            x = mmread(fhand)
        return x

    def get_text_data(self):
        with open(self.data_path('text_features.mtx'), 'r') as fhand:
            x = mmread(fhand)
        return x


def get_dat_data():
    # retrieve the datasets
    print('retrieving data')
    p = Processing()
    text_feats = p.get_text_data().tocsr()
    char_feats = p.get_char_data().tocsr()
    rest_feats = p.datamart['data']
    valid_rest_feats = p.datamart['valid']
    valid_index = valid_rest_feats.index
    p.datamart.close()
    y = rest_feats['SalaryNormalized'].values
    del rest_feats['SalaryNormalized']
    t_normer = StandardScaler()
    t_normer.fit(rest_feats.values)
    sp_rest_feats = sparse.csr_matrix(t_normer.transform(rest_feats.values))
    sp_rest_feats_valid = sparse.csr_matrix(t_normer.transform(valid_rest_feats.values))
    data_len = y.shape[0]
    x_dense = rest_feats.values
    x_dense_valid = valid_rest_feats.values
    x_sparse = text_feats[:data_len].tocsr()
    x_char = char_feats[:data_len]
    # generate train/test split
    print('shuffling data')
    inders_game = np.arange(data_len)
    np.random.seed(seed=123)
    np.random.shuffle(inders_game)
    print('data set size: %i' % data_len)
    test_size = .50
    cv = ShuffleSplit(data_len, 1, test_size=test_size, random_state=123)
    for tr, te in cv:
        train_ind = tr
        test_ind = te
    x_sparse_train = sparse.hstack((x_sparse[train_ind], sp_rest_feats[train_ind])).tocsr()
    x_dense_train = x_dense[train_ind]
    x_char_train = sparse.hstack((x_char[train_ind], sp_rest_feats[train_ind])).tocsr()
    y_train = y[train_ind]
    x_sparse_test = sparse.hstack((x_sparse[test_ind], sp_rest_feats[test_ind])).tocsr()
    x_dense_test = x_dense[test_ind]
    x_char_test = sparse.hstack((x_char[test_ind], sp_rest_feats[test_ind])).tocsr()
    y_test = y[test_ind]
    x_sparse_valid = sparse.hstack((text_feats[data_len:].tocsr(), sp_rest_feats_valid)).tocsr()
    x_char_valid = sparse.hstack((char_feats[data_len:], sp_rest_feats_valid)).tocsr()
    return x_dense_train, y_train, x_dense_test, x_dense_valid, \
        x_sparse_train, x_sparse_test, x_sparse_valid, y_test, x_char_train, \
        x_char_test, x_char_valid, valid_index


def run_all_models():
    x_dense_train, y_train, x_dense_test, x_dense_valid, x_sparse_train,\
        x_sparse_test, x_sparse_valid, y_test, x_char_train, x_char_test, \
        x_char_valid, valid_index \
        = get_dat_data()
    models_dense = [rf_model, gradient_model, svr_model, mlp_model,
                    gradient_model_lad, ada_model]
    models_new = [neighbors_model]
    all_test_pred = []
    all_valid_pred = []
    print('fitting fancy_text_model')
    ftext_test_pred, fvalid_pred = fancy_text_model(x_sparse_train, y_train,
                                                    x_sparse_test,
                                                    x_sparse_valid,
                                                    'text_model.pkl',
                                                    use_cache=True)
    print('MAE of fancy_text_model')
    print(mae(y_test, ftext_test_pred))
    all_test_pred.append(ftext_test_pred)
    all_valid_pred.append(fvalid_pred)
    print('fitting sgd_text_model')
    sgd_text_test_pred, sgd_text_valid_pred = sgd_text_model(x_sparse_train, y_train,
                                                             x_sparse_test,
                                                             x_sparse_valid,
                                                             'sgdtext_model.pkl',
                                                             use_cache=False)
    print('MAE of sgd_text_model')
    print(mae(y_test, sgd_text_test_pred))
    all_test_pred.append(sgd_text_test_pred)
    all_valid_pred.append(sgd_text_valid_pred)
    print('fitting fancy_text_model on char')
    fchar_test_pred, fchar_pred = fancy_text_model(x_char_train, y_train,
                                                   x_char_test,
                                                   x_char_valid,
                                                   'char_model.pkl',
                                                   use_cache=True)
    print('MAE of fancy_text_model on char')
    print(mae(y_test, fchar_test_pred))
    all_test_pred.append(fchar_test_pred)
    all_valid_pred.append(fchar_pred)
    sgd_char_test_pred, sgd_char_valid_pred = sgd_text_model(x_char_train, y_train,
                                                             x_char_test,
                                                             x_char_valid,
                                                             'sgdchar_model.pkl',
                                                             use_cache=False)
    print('MAE of sgd_text_model on char')
    print(mae(y_test, sgd_char_test_pred))
    all_test_pred.append(sgd_char_test_pred)
    all_valid_pred.append(sgd_char_valid_pred)
    normer = StandardScaler()
    x_dense_train = normer.fit_transform(x_dense_train)
    x_dense_test = normer.transform(x_dense_test)
    x_dense_valid = normer.transform(x_dense_valid)
    print(x_dense_valid.shape)
    for model in models_dense:
        print('fitting %s' % str(model.func_name))
        cache_name = 'full' + model.func_name + '.pkl'
        t_test_pred, t_valid_pred = model(x_dense_train, y_train,
                                          x_dense_test, x_dense_valid,
                                          cache_name, use_cache=True)
        print('MAE of %s' % str(model.func_name))
        print(mae(y_test, t_test_pred))
        all_test_pred.append(t_test_pred)
        all_valid_pred.append(t_valid_pred)
    for model in models_new:
        print('fitting %s' % str(model.func_name))
        cache_name = 'full' + model.func_name + '.pkl'
        t_test_pred, t_valid_pred = model(x_dense_train, y_train,
                                          x_dense_test, x_dense_valid,
                                          cache_name, use_cache=False)
        print('MAE of %s' % str(model.func_name))
        print(mae(y_test, t_test_pred))
        all_test_pred.append(t_test_pred)
        all_valid_pred.append(t_valid_pred)
    print('ensembling predictions')
    all_test_pred = np.vstack(all_test_pred).T
    all_valid_pred = np.vstack(all_valid_pred).T
    all_test_pred = normer.fit_transform(all_test_pred)
    all_valid_pred = normer.transform(all_valid_pred)
    test_pred, valid_pred = ensembler(all_test_pred, y_test, all_valid_pred)
    print('MAE of ensemble')
    print(mae(y_test, test_pred))
    write_soln(valid_pred, valid_index)


def sgd_text_model(x_train, y_train, x_test, x_valid, cache_name, use_cache=False):
    if use_cache:
        fhand = open(cache_name, 'r')
        data_dict = pickle.load(fhand)
        return data_dict['test_pred'], data_dict['valid_pred']
    np.random.seed(seed=123)
    model = SGDRegressor(eta0=1000, fit_intercept=True, l1_ratio=0.15,
                         learning_rate='invscaling', loss='huber', n_iter=200,
                         p=None, penalty='l1', power_t=.1, random_state=123,
                         rho=None, shuffle=True, verbose=0, warm_start=False)
    model.fit(x_train, y_train)
    test_pred = model.predict(x_test)
    valid_pred = model.predict(x_valid)
    data_dict = {'test_pred': test_pred, 'valid_pred': valid_pred}
    fhand = open(cache_name, 'w')
    pickle.dump(data_dict, fhand)
    fhand.close()
    return test_pred, valid_pred


def fancy_text_model(x_train, y_train, x_test, x_valid, cache_name, use_cache=False):
    if use_cache:
        fhand = open(cache_name, 'r')
        data_dict = pickle.load(fhand)
        return data_dict['test_pred'], data_dict['valid_pred']
    np.random.seed(seed=123)
    model = PassiveAggressiveRegressor(n_iter=100, C=1, shuffle=True, random_state=123)
    model.fit(x_train, y_train)
    test_pred = model.predict(x_test)
    valid_pred = model.predict(x_valid)
    data_dict = {'test_pred': test_pred, 'valid_pred': valid_pred}
    fhand = open(cache_name, 'w')
    pickle.dump(data_dict, fhand)
    fhand.close()
    return test_pred, valid_pred


def rf_model(x_train, y_train, x_test, x_valid, cache_name, use_cache=False):
    n = 2000
    njobs = -1
    if use_cache:
        fhand = open(cache_name, 'r')
        data_dict = pickle.load(fhand)
        return data_dict['test_pred'], data_dict['valid_pred']
    np.random.seed(seed=123)
    model = RandomForestRegressor(n_estimators=n, n_jobs=njobs,
                                  min_samples_split=25, random_state=123)
    model.fit(x_train, np.log(y_train))
    test_pred = np.exp(model.predict(x_test))
    valid_pred = np.exp(model.predict(x_valid))
    data_dict = {'test_pred': test_pred, 'valid_pred': valid_pred}
    fhand = open(cache_name, 'w')
    pickle.dump(data_dict, fhand)
    fhand.close()
    return test_pred, valid_pred


def neighbors_model(x_train, y_train, x_test, x_valid, cache_name, use_cache=False):
    model = KNeighborsRegressor(n_neighbors=25)
    if use_cache:
        fhand = open(cache_name, 'r')
        data_dict = pickle.load(fhand)
        return data_dict['test_pred'], data_dict['valid_pred']
    np.random.seed(seed=123)
    model.fit(x_train, np.log(y_train))
    test_pred = np.exp(model.predict(x_test))
    valid_pred = np.exp(model.predict(x_valid))
    data_dict = {'test_pred': test_pred, 'valid_pred': valid_pred}
    fhand = open(cache_name, 'w')
    pickle.dump(data_dict, fhand)
    fhand.close()
    return test_pred, valid_pred


def gradient_model(x_train, y_train, x_test, x_valid, cache_name, use_cache=False):
    n = 10000
    if use_cache:
        fhand = open(cache_name, 'r')
        data_dict = pickle.load(fhand)
        return data_dict['test_pred'], data_dict['valid_pred']
    np.random.seed(seed=123)
    model = GradientBoostingRegressor(n_estimators=n,
                                      min_samples_split=25,
                                      random_state=123)
    model.fit(x_train, np.log(y_train))
    test_pred = np.exp(model.predict(x_test))
    valid_pred = np.exp(model.predict(x_valid))
    data_dict = {'test_pred': test_pred, 'valid_pred': valid_pred}
    fhand = open(cache_name, 'w')
    pickle.dump(data_dict, fhand)
    fhand.close()
    return test_pred, valid_pred


def gradient_model_lad(x_train, y_train, x_test, x_valid, cache_name, use_cache=False):
    if use_cache:
        fhand = open(cache_name, 'r')
        data_dict = pickle.load(fhand)
        return data_dict['test_pred'], data_dict['valid_pred']
    np.random.seed(seed=123)
    model = GradientBoostingRegressor(n_estimators=500,
                                      loss='lad',
                                      min_samples_split=25,
                                      random_state=123,
                                      verbose=0)
    model.fit(x_train, np.log(y_train))
    test_pred = np.exp(model.predict(x_test))
    valid_pred = np.exp(model.predict(x_valid))
    data_dict = {'test_pred': test_pred, 'valid_pred': valid_pred}
    fhand = open(cache_name, 'w')
    pickle.dump(data_dict, fhand)
    fhand.close()
    return test_pred, valid_pred


def mlp_model(x_train, y_train, x_test, x_valid, cache_name, use_cache=False):
    nhidden = 40
    epochs = 2000
    if use_cache:
        fhand = open(cache_name, 'r')
        data_dict = pickle.load(fhand)
        return data_dict['test_pred'], data_dict['valid_pred']
    np.random.seed(seed=123)
    model = MLPRegressor(n_hidden=nhidden, batch_size=10000, use_dropout=False)
    model.fit(x_train, np.log(y_train).reshape(-1, 1),
              max_epochs=epochs)
    test_pred = np.exp(model.predict(x_test)).ravel()
    valid_pred = np.exp(model.predict(x_valid)).ravel()
    data_dict = {'test_pred': test_pred, 'valid_pred': valid_pred}
    fhand = open(cache_name, 'w')
    pickle.dump(data_dict, fhand)
    fhand.close()
    return test_pred, valid_pred


def svr_model(x_train, y_train, x_test, x_valid, cache_name, use_cache=False):
    if use_cache:
        fhand = open(cache_name, 'r')
        data_dict = pickle.load(fhand)
        return data_dict['test_pred'], data_dict['valid_pred']
    np.random.seed(seed=123)
    model = SVR()
    model.fit(x_train, np.log(y_train))
    test_pred = np.exp(model.predict(x_test))
    valid_pred = np.exp(model.predict(x_valid))
    data_dict = {'test_pred': test_pred, 'valid_pred': valid_pred}
    fhand = open(cache_name, 'w')
    pickle.dump(data_dict, fhand)
    fhand.close()
    return test_pred, valid_pred


def ada_model(x_train, y_train, x_test, x_valid, cache_name, use_cache=False):
    if use_cache:
        fhand = open(cache_name, 'r')
        data_dict = pickle.load(fhand)
        return data_dict['test_pred'], data_dict['valid_pred']
    np.random.seed(seed=123)
    model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(random_state=123, min_samples_split=25),
                              n_estimators=1000, loss='linear',
                              random_state=123)
    model.fit(x_train, np.log(y_train))
    test_pred = np.exp(model.predict(x_test))
    valid_pred = np.exp(model.predict(x_valid))
    data_dict = {'test_pred': test_pred, 'valid_pred': valid_pred}
    fhand = open(cache_name, 'w')
    pickle.dump(data_dict, fhand)
    fhand.close()
    return test_pred, valid_pred


def ensembler(x_test, y_test, x_valid):
    nestimators = 2000
    np.random.seed(seed=123)
    model = RandomForestRegressor(n_estimators=nestimators, n_jobs=-1,
                                  min_samples_split=25, oob_score=True,
                                  random_state=123)
    model.fit(x_test, y_test)
    valid_pred = model.predict(x_valid)
    return model.oob_prediction_, valid_pred


def write_soln(pred, index, fname='rando_model'):
    fhand = open('../data/%s.csv' % fname, 'w')
    fhand.write('Id, SalaryNormalized\n')
    for ind, x in zip(index, pred):
        fhand.write('%i, %f\n' % (ind, x))
    fhand.close()


def master_method():
    p = Processing()
    p.make_data()
    p.make_text_data()
    p.make_char_data()
    p.make_hash_data
    # %s/use_cache=True/use_cache=False/g
    run_all_models()


if __name__ == '__main__':
    run_all_models()
