from abc import abstractmethod

from myautoml.model.modelregistryholder import ModelRegistryHolder


class BaseModel(metaclass=ModelRegistryHolder):
    pass

    @abstractmethod
    def fit(self, x, y):
        pass
