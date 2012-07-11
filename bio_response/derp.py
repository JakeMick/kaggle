#!/usr/bin/env python
import numpy as np
import scipy as sp
import pandas
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import time

# Some settings
cores=8
bags=40
nClassifiers = 100
classifiers = [
        ExtraTreesClassifier(n_estimators=bags, n_jobs=cores, criterion='gini',
            bootstrap=True, oob_score=True),
        ExtraTreesClassifier(n_estimators=bags, n_jobs=cores, criterion='entropy',
            bootstrap=True, oob_score=True),
        RandomForestClassifier(n_estimators=bags, n_jobs=cores, criterion='gini',
            bootstrap=True, oob_score=True),
        RandomForestClassifier(n_estimators=bags, n_jobs=cores, criterion='entropy',
            bootstrap=True, oob_score=True)]

# Import the data
training = pandas.read_csv('train.csv')
testing = pandas.read_csv('test.csv')

# Sanity Check
assert( np.all(training.columns[1:] == testing.columns) )

# Make it a ndarray and remove training labels
trainingData = training.as_matrix()
xTrain = trainingData[:,1:]
yTrain = trainingData[:,0]

testingData = testing.as_matrix()

def logloss(act, pred):
    epsilon = 1e-4
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = -1.0/len(act) * sum(act*sp.log(pred) +
            sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    return ll


# Stores the outputs of out trees
trainingBag = []
testingBag = []

for _ in xrange(nClassifiers):
    for c in classifiers:
        c.fit(xTrain,yTrain)
        decisionFunc = c.oob_decision_function_[:,1]
        trainingBag.append(decisionFunc)
        testingBag.append(c.predict_proba(testingData)[:,1])

# makes the list of arrays into a matrix
trainingBag = np.vstack(trainingBag).T
testingBag = np.vstack(testingBag).T

# Grid searching over alpha

stratifiedCV = StratifiedKFold(yTrain,10)

ansDict = {}
ansDict['train'] = {}
ansDict['test'] = {}
for ind,a in enumerate(np.arange(-10,1,.1)):
    print(time.time())
    ansDict['train'][ind] = []
    ansDict['test'][ind] = []
    for train,test in stratifiedCV:
        X_train,X_test,Y_train,Y_test =\
                trainingBag[train,:],trainingBag[test,:],yTrain[train,:],yTrain[test,:]
        sgd = SGDClassifier(loss='log',penalty='l1',n_iter=10000,alpha=10**a)
        sgd.fit(X_train,Y_train)
        tempPred = sgd.predict_proba(X_test)
        ansDict['train'][ind].append(logloss(Y_test,tempPred))
        ansDict['test'][ind].append(sgd.predict_proba(testingBag))

meanScores = []
for i,j in ansDict['train'].items():
    wut = np.array(j)
    meanScores.append(wut.mean())

meanScores = np.array(meanScores)
meanScores[meanScores < 0] = 1.0
print(meanScores.min())
paramGood = np.where(meanScores == meanScores.min())[0][0]
testPred = ansDict['test'][paramGood]
finalPred = np.vstack(testPred).mean(axis=0)

def write_prediction(f):
    g = open('better_prediction.csv','w')
    for i in f:
        g.write(str(i)+'\n')
    g.close()

write_prediction(finalPred)



