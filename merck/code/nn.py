#! /usr/bin/env python
from ffnet import ffnet, mlgraph, tmlgraph

class OneHiddenLayerMomentum():
    def __init__(self, hidden_node=10, maxiter=100, eta=0.2):
        self.hidden_node=hidden_node
        self.maxiter=maxiter
        self.eta=eta
    def fit(self, training_set, training_target):
        self.feature_size = training_set.shape[1]
        hidden_layer_size = self.hidden_node
        connection_tuple = (self.feature_size, hidden_layer_size, 1)
        self.nn = ffnet(mlgraph(connection_tuple))
        self.nn.train_momentum(training_set, training_target)
    def predict(self, testing_set):
        return self.nn.call(testing_set)
