from myautoml.estimator.baseestimator import BaseEstimator
from sklearn.metrics import mean_squared_error

class AutoMLRegressor(BaseEstimator):

    def __init__(self) -> None:
        super().__init__(estimator = 'regressor',
                         metric=mean_squared_error)


    def fit(self, dataset):
        pass

    def predict(self, dataset):
        pass
