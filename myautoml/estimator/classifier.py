
from myautoml.estimator.baseestimator import BaseEstimator

from sklearn.metrics import roc_auc_score

class Classifier(BaseEstimator):

    ESTIMATOR_TYPE = 'classifier'

    def __init__(self) -> None:
        from myautoml import class_models
        super().__init__(estimator = self.ESTIMATOR_TYPE,
                         metric=roc_auc_score,
                         model=class_models['DecisionTree'])

    def fit(self, X_train=None, y_train=None):
        print('start fitting model classifier')
        self.model.fit(X_train, y_train)

    def predict(self, X_test=None):
        print('start predicting')
        return self.model.predict(X_test)