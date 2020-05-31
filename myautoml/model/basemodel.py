from abc import abstractmethod

from pandas import DataFrame, Series

from myautoml.model.modelregistryholder import ModelRegistryHolder


class BaseModel(metaclass=ModelRegistryHolder):


    def __init__(self):
        self.model = None
        self._is_fit_done = False


    @abstractmethod
    def fit(self, x:DataFrame, y:Series):
        ##do smth
        self.model.fit(x, y)
        # call whatever we want and config if needed
        self._is_fit_done = True


    @abstractmethod
    def predict(self, x:DataFrame):
        if not self._is_fit_done:
            raise Exception("Call fit() before call predict()")
        return self.model.predict(x)

    @abstractmethod
    def _init_model(self):
        pass
