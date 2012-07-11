#!/usr/bin/env python
import numpy as np
import pandas
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
import pickle
from os.path import isfile
from sklearn.svm import SVC
from sklearn.preprocessing import Scaler
from logloss import logloss


def modelg():
# Some settings
    model = GradientBoostingClassifier(loss='deviance', subsample=.5, n_estimators=100000)
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

    stratifiedCV = StratifiedKFold(yTrain,10)

    scores = []
    pred = []
    for train,test in stratifiedCV:
        X_train,X_test,Y_train,Y_test =\
                xTrain[train,:],xTrain[test,:],yTrain[train,:],yTrain[test,:]
        model.fit(X_train,Y_train)
        accur = logloss(Y_test,model.predict_proba(X_test)[:,1])
        scores.append(accur)
        print(accur)
        pred.append(model.predict_proba(testingData)[:,1])

    meanScores = np.array(scores)
    print(meanScores.mean())
    finalPred = np.vstack(pred).mean(axis=0)

    def write_prediction(f):
        g = open('grad_prediction.csv','w')
        for i in f:
            g.write(str(i)+'\n')
        g.close()

    write_prediction(finalPred)

class model2(object):
    def __init__(self):
        self.load_processed_data()
    def grid_searcher(self):
        X_train, X_test, Y_train, Y_test = self.cv_data[-1]
        X_train = np.vstack((X_train,X_test))
        Y_train = np.concatenate((Y_train,Y_test))
        stratifiedCV = StratifiedKFold(Y_train,10)

        ansDict = {}
        ansDict['train'] = {}
        ansDict['test'] = {}

        C_range = 10.0 ** np.arange(-4, 9)
        gamma_range = 10.0 ** np.arange(-5,4)
        for ind, i in enumerate(C_range):
            for jnd, j  in enumerate(gamma_range):
                # Cantor's pairs
                dictInd = ((ind + jnd + 2)**2 + (ind + 1) - (jnd + 1))/2
                ansDict['train'][dictInd] = []
                ansDict['test'][dictInd] = []
                for train,test in stratifiedCV:
                    X_trainT, X_testT, Y_trainT, Y_testT = X_train[train,:],\
                            X_train[test,:],Y_train[train,:],Y_train[test,:]
                    svc = SVC(kernel='rbf',C=i, gamma=j, probability=True,class_weight='auto')
                    svc.fit(X_trainT,Y_trainT)
                    ansDict['train'][dictInd].append(logloss(Y_trainT,svc.predict_proba(X_trainT)[:,1]))
                    ansDict['test'][dictInd].append(svc.predict_proba(self.testMat)[:,1])

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
            g = open('sc_prediction.csv','w')
            for i in f:
                g.write(str(i)+'\n')
            g.close()

        write_prediction(finalPred)


    def balance_data(self,training,trainingAns):
        labels = np.unique(trainingAns.astype(int))
        observationsLen = trainingAns.shape[0]
        if (trainingAns == labels[0]).sum() < observationsLen/2.0:
            minorityLabel = labels[0]
        else:
            minorityLabel = labels[1]
        trainInd = np.where(trainingAns == minorityLabel)[0]
        how_many = observationsLen - (2 * trainInd.shape[0])
        while trainInd.shape[0] < 10*how_many:
            trainInd = np.append(trainInd,trainInd)
        np.random.shuffle(trainInd)
        addedResult = trainingAns[trainInd[:how_many]]
        addedData = training[trainInd[:how_many],:]
        trainResult = np.append(trainingAns,addedResult)
        trainMat = np.vstack((training,addedData))
        return trainMat, trainResult


    def process_data(self):
        test = pandas.read_csv('test.csv')
        testMat = test.as_matrix()

        train = pandas.read_csv('train.csv')
        trainMat = train.as_matrix()
        trainResult = trainMat[:,0]
        trainMat = trainMat[:,1:]

        #trainInd = np.where(trainResult == 0)[0]
        #how_many = (trainResult == 1).sum() - len(trainInd)
        #np.random.shuffle(trainInd)
        #addedResult = trainResult[trainInd[:how_many],:]
        #addedData = trainMat[trainInd[:how_many],:]
        #trainResult = np.append(trainResult,addedResult)
        #trainMat = np.vstack((trainMat,addedData))

        cv = StratifiedKFold(trainResult,2)
        #cv = KFold(n=trainResult.shape[0],k=2)
        reduceFeatures = ExtraTreesClassifier(compute_importances=True, random_state=1234,n_jobs=self.cpus,n_estimators=1000, criterion='gini')
        reduceFeatures.fit(trainMat,trainResult)
        trainScaler = Scaler()

        self.cv_data = []
        self.cv_data_nonreduced = []
        for train, test in cv:
            X_train, X_test, Y_train, Y_test = trainMat[train,:], trainMat[test,:], trainResult[train,:], trainResult[test,:]
            X_train = trainScaler.fit_transform(X_train)
            X_test = trainScaler.transform(X_test)
            self.cv_data_nonreduced.append((X_train,X_test,Y_train,Y_test))
            X_train = reduceFeatures.transform(X_train)
            X_test = reduceFeatures.transform(X_test)
            self.cv_data.append((X_train,X_test,Y_train,Y_test))
        testMat = trainScaler.transform(testMat)
        self.testMat_nonreduced = testMat
        self.testMat = reduceFeatures.transform(testMat)
        allData = self.testMat, self.cv_data, self.testMat_nonreduced, self.cv_data_nonreduced
        data_handle = open('allData.pkl','w')
        pickle.dump(allData,data_handle)
        data_handle.close()

    def load_processed_data(self):
        if isfile('allData.pkl'):
            data_handle = open('allData.pkl','r')
            self.testMat, self.cv_data, self.testMat_nonreduced, self.cv_data_nonreduced = pickle.load(data_handle)
            data_handle.close()
        else:
            print("You need to run process_data() first.")

def blend_models():
    model_csv = ['best_prediction.csv','sc_prediction.csv']
    models = []
    for m in model_csv:
        tempCsv = pandas.read_csv(m,header=None)
        models.append(tempCsv.as_matrix())
    return models



