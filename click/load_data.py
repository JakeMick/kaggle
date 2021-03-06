import numpy as np
import pandas
from ml_metrics import rmse
from sklearn.base import BaseEstimator, RegressorMixin, MetaEstimatorMixin, clone
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import Bootstrap
from sklearn.linear_model import PassiveAggressiveRegressor, Ridge
from sklearn.svm import NuSVR
from sklearn.neighbors import KNeighborsRegressor


class loader:
    def __init__(self, n_text=100):
        self.inputs = ['latitude', 'longitude', 'summary', 'description',
                       'source', 'created_time', 'tag_type']
        self.text = ['summary']
        self.targets = ['num_votes', 'num_comments', 'num_views']
        self.read()
        self.add_dates()
        self.add_geo()
        self.add_texts(n_text)
        self.format()

    def format(self):
        self.train_ind = self.train['id']
        del self.train['id']
        self.test_ind = self.test['id']
        del self.test['id']
        for i in self.test.columns:
            if self.test[i].dtype == 'object':
                del self.test[i]
        self.X_train = self.train[self.test.columns].values
        self.y_train = self.train[self.targets].values
        self.X_test = self.test.values

    def read(self):
        self.train = pandas.read_csv('train.csv', parse_dates=[9])
        self.test = pandas.read_csv('test.csv', parse_dates=[6])
        self.zip = pandas.read_csv('free-zipcode-database.csv')
        self.latlong()

    def latlong(self):
        self.zip = self.zip[self.zip.Lat.isnull() == False]
        self.zip = self.zip[['State', 'Lat', 'Long', 'TaxReturnsFiled', 'EstimatedPopulation', 'TotalWages']]
        self.zip['EstWages'] = self.zip.TotalWages / self.zip.TaxReturnsFiled
        self.zip['DependencyRatio'] = self.zip.TaxReturnsFiled / self.zip.EstimatedPopulation
        self.zip = self.zip.dropna()

    def add_geo(self):
        model = KNeighborsRegressor(n_neighbors=1)
        x = self.zip[['Lat', 'Long']].values
        derived = ['TaxReturnsFiled', 'EstimatedPopulation', 'EstWages', 'DependencyRatio']
        y = self.zip[derived].values
        model.fit(x, y)
        train_feats = model.predict(self.train[['latitude', 'longitude']].values)
        test_feats = model.predict(self.test[['latitude', 'longitude']].values)
        tr = pandas.DataFrame(train_feats, columns=derived)
        te = pandas.DataFrame(test_feats, columns=derived)
        self.merge(tr, self.train)
        self.merge(te, self.test)

    def add_dates(self):
        train_dates = self.date_trick(self.train.created_time)
        test_dates = self.date_trick(self.test.created_time)
        self.merge(train_dates, self.train)
        self.merge(test_dates, self.test)
        del self.test['created_time']

    def merge(self, left, right):
        for i in left.columns:
            right[i] = left[i]

    def add_texts(self, n_text):
        for i in self.text:
            count = CountVectorizer(max_features=n_text)
            count.fit(self.train[i])
            train_feats = count.transform(self.train[i]).toarray()
            test_feats = count.transform(self.test[i]).toarray()
            fnames = count.get_feature_names()
            cnames = map(lambda k: "Counts_" + i + "_" + k, fnames)
            tr = pandas.DataFrame(train_feats, columns=cnames)
            te = pandas.DataFrame(test_feats, columns=cnames)
            self.merge(tr, self.train)
            self.merge(te, self.test)
            del self.train[i]
            del self.test[i]

    def date_trick(self, series):
        inder = lambda i: np.linspace(0, 2 * np.pi, i)
        hour_ind = inder(24)
        week_ind = inder(7)
        month_ind = inder(32)
        representer = lambda i: np.vstack((np.sin(i), np.cos(i))).T
        hour, week, month = representer(hour_ind), representer(week_ind), representer(month_ind)
        out = np.zeros((series.shape[0], 6), dtype='float')
        for i in xrange(series.shape[0]):
            out[i, 0:2] = hour[series[i].hour]
            out[i, 2:4] = week[series[i].dayofweek]
            out[i, 4:6] = month[series[i].day]
        return pandas.DataFrame(out, columns=['Hour_i', 'Hour_r', 'Day_i', 'Day_r', 'Week_i', 'Week_r'])


def rmse_est(estimator, x, y):
    pred = estimator.predict(x)
    return -rmse(pred, y)


class process:
    def __init__(self, fname, n=150000, n_text=100):
        self.fname = fname
        data = loader(n_text=n_text)
        self.test_ind = data.test_ind
        self.X_tr = data.X_train[-n:]
        self.X_te = data.X_test
        self.y_tr = np.log(data.y_train[-n:] + 1)
        sc = StandardScaler()
        self.X_tr = sc.fit_transform(self.X_tr)
        self.X_te = sc.transform(self.X_te)
        self.is_classification = False

    def write_res(self, pred):
        out = np.exp(pred) - 1
        out[out < 0] = 0.0
        h = open(self.fname, 'w')
        h.write("id,num_votes,num_comments,num_views\n")
        for row in xrange(out.shape[0]):
            h.write("%i, %2.7f, %2.7f, %2.7f\n" % (self.test_ind[row],
                                                   pred[row][0],
                                                   pred[row][1],
                                                   pred[row][2]))
        h.close()

    def search_params(self, estimator, param_dict, n_iter=50, train_size=30000):
        self.pdict = param_dict
        self.est = estimator
        self.sampler = Bootstrap(self.X_tr.shape[0], n_iter=4, train_size=train_size)
        self.sampler.is_classification = False
        self.cv = RandomizedSearchCV(self.est, self.pdict, scoring=rmse_est,
                                     n_jobs=4, cv=self.sampler, refit=False, verbose=9,
                                     n_iter=n_iter)
        self.cv.is_classification = False
        self.cv.fit(self.X_tr, self.y_tr)
        print("____Best Score____")
        print(-self.cv.best_score_)
        print(self.cv.best_params_)


class SuperRegressor(BaseEstimator, RegressorMixin, MetaEstimatorMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **params):
        self.estimators_ = fit_multi(self.estimator, X, y)
        return self

    def predict(self, X):
        return predict_multi(self.estimators_, X)

    def set_params(self, **params):
        self.estimator.set_params(**params)


def fit_multi(estimator, X, y, **params):
    outputs = y.shape[1]
    estimators = []
    for i in xrange(outputs):
        est = clone(estimator)
        est.fit(X, y[:, i], **params)
        estimators.append(est)
    return estimators


def predict_multi(estimators, X):
    outputs = []
    for est in estimators:
        outputs.append(est.predict(X))
    return np.vstack(outputs).T


class RFmodel:
    def __init__(self):
        """
        n = 50000 {
            'max_features': 0.9,
            'min_samples_split': 14,
            'n_estimators': 900,
            'max_depth': None,
            'min_samples_leaf': 3}
        CV: 0.386739031894, Board: 0.40945
        'pred_rf_round1.csv'

        n = 150000 {
            'max_features': 'auto',
            'min_samples_split': 27,
            'n_estimators': 1950,
            'max_depth': None,
            'min_samples_leaf': 3}
        CV: 0.387281301019, Board, 0.41016
        'pred_rf_round2.csv'

        n = 50000 {
            'max_features': 'auto',
            'min_samples_split': 27,
            'n_estimators': 1950,
            'max_depth': None,
            'min_samples_leaf': 3}
        CV: ???, Board  0.40629,
        'pred_rf_round3.csv'

        n = 100000 {
            'max_features': 'auto',
            'min_samples_split': 27,
            'n_estimators': 1950,
            'max_depth': None,
            'min_samples_leaf': 3}
        CV: ???, Board:  0.40418,
        'pred_rf_round4.csv'


        """
        np.random.seed(123)
        self.love = process("pred_rf_round5.csv", n=100000, n_text=1000)
        self.make_model()

    def make_model(self):
        model = RandomForestRegressor(max_features='auto',
                                      min_samples_split=27,
                                      n_estimators=3000,
                                      max_depth=None,
                                      min_samples_leaf=3,
                                      n_jobs=8,
                                      verbose=9)
        model.fit(self.love.X_tr, self.love.y_tr)
        pred = model.predict(self.love.X_te)
        self.love.write_res(pred)

    def find_model(self):
        np.random.seed(123)
        model = RandomForestRegressor(max_depth=None, verbose=9, n_jobs=4)
        params = {'n_estimators': np.arange(1000, 2000, 50),
                  'max_features': [.9, .95, 'auto', 'sqrt', 'log2'],
                  'min_samples_split': np.arange(10, 100),
                  'min_samples_leaf': np.arange(1, 10),
                  }
        self.love.search_params(model, params, n_iter=100, train_size=30000)


class PAmodel:
    def __init__(self):
        np.random.seed(123)
        self.love = process("pred_pa_round2.csv", n=100000, n_text=1000)
        self.find_model()

    def make_model(self):
        model = SuperRegressor(PassiveAggressiveRegressor())
        model.fit(self.love.X_tr, self.love.y_tr)
        pred = model.predict(self.love.X_te)
        self.love.write_res(pred)

    def find_model(self):
        np.random.seed(123)
        model = SuperRegressor(PassiveAggressiveRegressor())
        params = {
            'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            'C': np.arange(.1, 1.1, .1),
            'epsilon': np.arange(0.01, .5, .1),
            'n_iter': np.arange(100, 10000, 20),
        }
        self.love.search_params(model, params, n_iter=10, train_size=30000)


class SVMmodel:
    def __init__(self):
        np.random.seed(123)
        self.love = process("pred_svm_round1.csv", n=100000)
        self.find_model()

    def make_model(self):
        model = SuperRegressor(NuSVR(random_state=123))
        model.fit(self.love.X_tr, self.love.y_tr)
        pred = model.predict(self.love.X_te)
        self.love.write_res(pred)

    def find_model(self):
        np.random.seed(123)
        model = SuperRegressor(NuSVR(random_state=123))
        params = {
            'C': np.arange(.1, 5.1, .1),
            'nu': np.arange(.01, 1.01, .1),
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'degree': np.arange(1, 6),
            'gamma': np.arange(0, 1, .1),
            'shrinking': [False, True],
        }
        self.love.search_params(model, params, n_iter=10, train_size=30000)


class RidgeModel:
    def __init__(self):
        np.random.seed(123)
        self.love = process("pred_ridge_round4.csv", n=70000, n_text=1000)
        self.make_model()

    def make_model(self):
        model = SuperRegressor(Ridge(solver='lsqr', alpha=3.88))
        model.fit(self.love.X_tr, self.love.y_tr)
        pred = model.predict(self.love.X_te)
        pred[pred < 0.0] = 0.0
        self.love.write_res(pred)

    def find_model(self):
        np.random.seed(123)
        model = SuperRegressor(Ridge(solver='lsqr'))
        params = {
            'alpha': np.arange(.01, 5.1, .01),
        }
        self.love.search_params(model, params, n_iter=100, train_size=30000)


class GBMModel:
    def __init__(self):
        np.random.seed(123)
        self.love = process("pred_gbm_round1.csv", n=27000, n_text=1000)
        self.make_model()

    def make_model(self):
        model = SuperRegressor(GradientBoostingRegressor())
        model.fit(self.love.X_tr, self.love.y_tr)
        pred = model.predict(self.love.X_te)
        pred[pred < 0.0] = 0.0
        self.love.write_res(pred)

    def find_model(self):
        np.random.seed(123)
        model = SuperRegressor(GradientBoostingRegressor(loss='ls'))
        params = {
                  'n_estimators': np.arange(1000, 4000, 50),
                  'learning_rate': np.arange(0.001, .3, .001),
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'min_samples_split': np.arange(10, 100),
                  'subsample': np.arange(0.01, 1.01, .01),
                  }
        self.love.search_params(model, params, n_iter=200, train_size=10000)


def blender():
    files = ["pred_rf_round5.csv","pred_ridge_round4.csv"]
    n = float(len(files))
    new = pandas.read_csv(files[0])
    new.index = new.id
    del new['id']
    data = []
    for i in files:
        frame = pandas.read_csv(i)
        frame.index = frame.id
        assert(np.all(new.index == frame.index))
        data.append(frame)
    new.num_votes = sum([i.num_votes for i in data]) / n
    new.num_views = sum([i.num_views for i in data]) / n
    new.num_comments = sum([i.num_comments for i in data]) / n
    new[new < 0.001] = 0.0
    new.to_csv("pred_blended.csv")
