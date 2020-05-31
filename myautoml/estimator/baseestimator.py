from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    estimator = None  # class | reg
    metric = None
    model = None

    def __init__(self,
                 estimator=None,
                 metric=None,
                 model=None) -> None:
        self.model = model
        self.metric = metric
        self.estimator = estimator

    @abstractmethod
    def fit(self, X_train=None, y_train=None):
        pass

    @abstractmethod
    def predict(self, X_test=None):
        pass
