import pickle
from os.path import isfile
import numpy as np
import pandas
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import SGDClassifier, BayesianRidge, LassoLars, \
        ElasticNet, OrthogonalMatchingPursuit, LinearRegression, ARDRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.preprocessing import Scaler
from sklearn.grid_search import GridSearchCV
from logloss import logloss

class bagged_model(object):
    """ Before you run me:
    mkdir model1 through modeln
    download the train and test csv
    get a logloss function in your other

    tested on sklearn-dev
    for <8gb ram or laptops
        decrease self.bags, and model6 + blend_models' n_estimators
    """
    def __init__(self):
        self.cpus = 8
        self.bags = 5000
        self.load_processed_data()

    def grid_searcher(self):
        X_train, X_test, Y_train, Y_test = self.cv_data[-1]
        X_train = np.vstack((X_train,X_test))
        Y_train = np.concatenate((Y_train,Y_test))
        C_range = 10.0 ** np.arange(-2, 9,.5)
        gamma_range = 2.0 ** np.arange(-13,0,.5)
        param_grid = dict(gamma=gamma_range, C=C_range)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(y=Y_train, k=10), n_jobs=4, loss_func=logloss)
        grid.fit(X_train, Y_train)
        print("The best classifier is: ", grid.best_estimator_)
        return grid.predict_proba(self.testMat)

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

    def fit_model_1(self, lol = 0.0025, toWrite = False):
        model = SVC(probability = True, kernel = 'rbf', tol = 1e-3, gamma = 0.001, coef0 = 0.0)

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            model.fit(X_train,Y_train)
            pred = model.predict_proba(X_test)[:,1]
            print("Model 1 score: %f" % (logloss(Y_test,pred),))
        if toWrite:
            f2 = open('model1/model.pkl','w')
            pickle.dump(model,f2)

            #trainHandle = open('trainList.pkl','w')
            #pickle.dump(trainList,trainHandle)
            #trainHandle.close()
            #testHandle = open('testList.pkl','w')
            #pickle.dump(testList,testHandle)
            #testHandle.close()

    def fit_model_2(self, lol = .07, toWrite = False):
        model = LogisticRegression(C = lol, penalty = 'l1', tol = 1e-6)

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            X_train,Y_train = self.balance_data(X_train,Y_train)
            model.fit(X_train,Y_train)
            pred = model.predict_proba(X_test)[:,1]
            print("Model 2 Score: %f" % (logloss(Y_test,pred),))

        if toWrite:
            f2 = open('model2/model.pkl','w')
            pickle.dump(model,f2)
            f2.close()

    def fit_model_3(self, lol = 0.011,toWrite = True):
        model = SGDClassifier(penalty = 'l1', loss = 'log', n_iter = 50000, alpha = lol)

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            X_train,Y_train = self.balance_data(X_train,Y_train)
            model.fit(X_train,Y_train)
            pred = model.predict_proba(X_test)
            print("Model 3 score : %f" % (logloss(Y_test,pred),))

        if toWrite:
            f2 = open('model3/model.pkl','w')
            pickle.dump(model,f2)
            f2.close()

    def fit_model_4(self,toWrite=False):
        model = SVC(kernel='poly',probability=True, degree=2)

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            model.fit(X_train,Y_train)
            pred = model.predict_proba(X_test)[:,1]
            print("Model 4 score %f" % (logloss(Y_test,pred),))

        if toWrite:
            f2 = open('model4/model.pkl','w')
            pickle.dump(model,f2)
            f2.close()

    def fit_model_6(self,toWrite=False):
        model = RandomForestClassifier(n_estimators=2000,n_jobs=self.cpus)

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            X_train,Y_train = self.balance_data(X_train,Y_train)
            model.fit(X_train,Y_train)
            pred = model.predict_proba(X_test)[:,1]
            print("Model 6 score %f" % (logloss(Y_test,pred),))

        if toWrite:
            f2 = open('model6/model.pkl','w')
            pickle.dump(model,f2)
            f2.close()

    def fit_model_7(self,toWrite=False):
        model = NuSVC(probability=True,kernel='linear')

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            model.fit(X_train,Y_train)
            pred = model.predict_proba(X_test)[:,1]
            print("Model 7 score %f" % (logloss(Y_test,pred),))

        if toWrite:
            f2 = open('model7/model.pkl','w')
            pickle.dump(model,f2)
            f2.close()

    def fit_model_8(self,lol = 0.0, toWrite=False):
        model = BernoulliNB(alpha = lol, binarize = 0.0)

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            model.fit(X_train,Y_train)
            pred = model.predict_proba(X_test)[:,1]
            print("Model 8 score %f" % (logloss(Y_test,pred),))

        if toWrite:
            f2 = open('model8/model.pkl','w')
            pickle.dump(model,f2)
            f2.close()

    def fit_model_9(self,toWrite=False):
        model = GaussianNB()

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            model.fit(X_train,Y_train)
            pred = model.predict_proba(X_test)[:,1]
            print("Model 9 score %f" % (logloss(Y_test,pred),))

        if toWrite:
            f2 = open('model9/model.pkl','w')
            pickle.dump(model,f2)
            f2.close()

    def fit_model_10(self,toWrite=False):
        model = BayesianRidge(n_iter=5000)

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            model.fit(X_train,Y_train)
            pred = model.predict(X_test)
            print("Model 10 score %f" % (logloss(Y_test,pred),))

        if toWrite:
            f2 = open('model10/model.pkl','w')
            pickle.dump(model,f2)
            f2.close()

    def fit_model_11(self,toWrite=False):
        model = LassoLars(alpha=1,max_iter=5000)

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            model.fit(X_train,Y_train)
            pred = model.predict(X_test)
            print("Model 11 score %f" % (logloss(Y_test,pred),))

        if toWrite:
            f2 = open('model11/model.pkl','w')
            pickle.dump(model,f2)
            f2.close()

    def fit_model_12(self,toWrite=False):
        model = ElasticNet(alpha=1.0)

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            model.fit(X_train,Y_train)
            pred = model.predict(X_test)
            print("Model 12 score %f" % (logloss(Y_test,pred),))

        if toWrite:
            f2 = open('model12/model.pkl','w')
            pickle.dump(model,f2)
            f2.close()

    def fit_model_14(self,toWrite=False):
        model = OrthogonalMatchingPursuit()

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            model.fit(X_train,Y_train)
            pred = model.predict(X_test)
            print("Model 14 score %f" % (logloss(Y_test,pred),))

        if toWrite:
            f2 = open('model14/model.pkl','w')
            pickle.dump(model,f2)
            f2.close()

    def fit_model_15(self,toWrite=False):
        model = LinearRegression()

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            model.fit(X_train,Y_train)
            pred = model.predict(X_test)
            print("Model 15 score %f" % (logloss(Y_test,pred),))

        if toWrite:
            f2 = open('model15/model.pkl','w')
            pickle.dump(model,f2)
            f2.close()

    def fit_model_16(self,toWrite=False):
        model = ARDRegression()

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            model.fit(X_train,Y_train)
            pred = model.predict(X_test)
            print("Model 16 score %f" % (logloss(Y_test,pred),))

        if toWrite:
            f2 = open('model16/model.pkl','w')
            pickle.dump(model,f2)
            f2.close()

    def fit_model_18(self, lol = 0.0025, toWrite = False):
        model = SVC(probability = True, kernel = 'rbf', class_weight = 'auto', tol = 1e-3)

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            model.fit(X_train,Y_train)
            pred = model.predict_proba(X_test)[:,1]
            print("Model 18 score: %f" % (logloss(Y_test,pred),))
        if toWrite:
            f2 = open('model18/model.pkl','w')
            pickle.dump(model,f2)

    def fit_model_19(self, lol = 2, toWrite = False):
        model = SVC(probability = True, kernel = 'poly', class_weight = 'auto', tol = 1e-3, degree = lol)

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            model.fit(X_train,Y_train)
            pred = model.predict_proba(X_test)[:,1]
            print("Model 19 score: %f" % (logloss(Y_test,pred),))
        if toWrite:
            f2 = open('model19/model.pkl','w')
            pickle.dump(model,f2)

    def fit_model_20(self, lol = 0.0025, toWrite = False):
        model = SVC(probability = True, kernel = 'linear', class_weight = 'auto', tol = 1e-3)

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            model.fit(X_train,Y_train)
            pred = model.predict_proba(X_test)[:,1]
            print("Model 20 score: %f" % (logloss(Y_test,pred),))
        if toWrite:
            f2 = open('model20/model.pkl','w')
            pickle.dump(model,f2)

    def fit_model_21(self, lol = 3, toWrite = False):
        model = SVC(probability = True, kernel = 'sigmoid', class_weight = 'auto', tol = 1e-3, coef0 = lol)

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            model.fit(X_train,Y_train)
            pred = model.predict_proba(X_test)[:,1]
            print("Model 21 score: %f" % (logloss(Y_test,pred),))
        if toWrite:
            f2 = open('model21/model.pkl','w')
            pickle.dump(model,f2)

    def fit_model_22(self, lol = 2, toWrite = False):
        model = SVC(probability = True, kernel = 'sigmoid', tol = 1e-3, coef0 = lol)

        for data in self.cv_data:
            X_train, X_test, Y_train, Y_test = data
            model.fit(X_train,Y_train)
            pred = model.predict_proba(X_test)[:,1]
            print("Model 22 score: %f" % (logloss(Y_test,pred),))
        if toWrite:
            f2 = open('model22/model.pkl','w')
            pickle.dump(model,f2)

    def blend_models(self):
        folders = [
                'model1', 'model2', 'model3', 'model4',
                'model6','model7','model8','model9',
                'model10','model11','model12', 'model14',
                'model15','model16','model18',
                'model19','model20', 'model21', 'model22']
        predict_insteads = (8,9,10,11,12,13)

        models = []
        for folder in folders:
            model_hand = open(folder+'/'+'model.pkl','r')
            models.append(pickle.load(model_hand))
            model_hand.close()

        for derp in self.cv_data:
            X_train, X_test, Y_train, Y_test = derp
            trainLen = Y_test.shape[0]
            modelLen = len(models)
            testLen = self.testMat.shape[0]

            trainBag = np.zeros([trainLen,modelLen],dtype=float)
            testBag = np.zeros([testLen,modelLen],dtype=float)

            for i in xrange(modelLen):
                model = models[i]
                if i in predict_insteads:
                    model.predict_proba = model.predict
                trainPred = model.predict_proba(X_test)
                testPred = model.predict_proba(self.testMat)
                if len(trainPred.shape) > 1:
                    trainPred = trainPred[:,1]
                    testPred = testPred[:,1]
                trainBag[:,i] = trainPred
                testBag[:,i] = testPred
            rf = ExtraTreesClassifier(n_estimators=1000,n_jobs=self.cpus, oob_score=True, bootstrap=True,criterion='gini')
            rf.fit(trainBag,Y_test)
            print("Final score is %f" %(logloss(Y_test,rf.oob_decision_function_[:,1])))
        test_final = rf.predict_proba(testBag)[:,1]
        pred_hand = open('prediction.csv','w')
        for row in test_final:
            pred_hand.write(str(row)+'\n')
        pred_hand.close()

    def reRunAll(self):
        """ Except preprocessing """
        self.load_processed_data()
        self.fit_model_1(toWrite=True)
        self.fit_model_2(toWrite=True)
        self.fit_model_3(toWrite=True)
        self.fit_model_4(toWrite=True)
        #self.fit_model_5(toWrite=True)
        self.fit_model_6(toWrite=True)
        self.fit_model_7(toWrite=True)
        self.fit_model_8(toWrite=True)
        self.fit_model_9(toWrite=True)
        self.fit_model_10(toWrite=True)
        self.fit_model_11(toWrite=True)
        self.fit_model_12(toWrite=True)
        #self.fit_model_13(toWrite=True)
        self.fit_model_14(toWrite=True)
        self.fit_model_15(toWrite=True)
        self.fit_model_16(toWrite=True)
        #self.fit_model_17(toWrite=True)
        self.fit_model_18(toWrite=True)
        self.fit_model_19(toWrite=True)
        self.fit_model_20(toWrite=True)
        self.blend_models()

