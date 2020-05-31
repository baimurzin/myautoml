from pandas import DataFrame, Series

from myautoml.model.basemodel import BaseModel
from sklearn.linear_model import LogisticRegression


class LogReg(BaseModel):


    def fit(self, x: DataFrame, y: Series):
        print('start fitting')
        super().fit(x, y)

    def predict(self, x: DataFrame):
        res = super().predict(x)
        #do smth special for this class
        return res

    def _init_model(self):
        self.model = LogisticRegression(
                penalty='l2', dual=False, tol=0.0001, C=1.0,
                fit_intercept=True, intercept_scaling=1, class_weight=None,
                random_state=0, solver='lbfgs', max_iter=100,
                multi_class='ovr', verbose=0, warm_start=False, n_jobs=-1)
        print(self.__class__.__name__ + ' model initialized')