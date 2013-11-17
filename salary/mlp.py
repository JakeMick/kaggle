import numpy as np
import warnings

from itertools import cycle, izip

from sklearn.utils import gen_even_slices
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelBinarizer


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

    def fit(self, X, y, max_epochs, shuffle_data, staged_sample=None, verbose=0):
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


class MLPClassifier(BaseMLP, ClassifierMixin):
    """ Multilayer Perceptron Classifier.

    Uses a neural network with one hidden layer.


    Parameters
    ----------


    Attributes
    ----------

    Notes
    -----


    References
    ----------"""
    def __init__(self, n_hidden=200, lr=0.1, l2decay=0, loss='cross_entropy',
                 output_layer='softmax', batch_size=100, use_dropout=False,
                 dropout_fraction=0.5, verbose=0):
        super(MLPClassifier, self).__init__(n_hidden, lr, l2decay, loss,
                                            output_layer, batch_size, use_dropout,
                                            dropout_fraction, verbose)

    def fit(self, X, y, max_epochs=10, shuffle_data=True, staged_sample=None):
        self.lb = LabelBinarizer()
        one_hot_labels = self.lb.fit_transform(y)
        super(MLPClassifier, self).fit(
            X, one_hot_labels, max_epochs,
            shuffle_data, staged_sample)
        return self

    def predict(self, X):
        prediction = super(MLPClassifier, self).predict(X)
        return self.lb.inverse_transform(prediction)


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
    def __init__(self, n_hidden=200, lr=0.1, l2decay=0, loss='square',
                 output_layer='linear', batch_size=100, use_dropout=True,
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
