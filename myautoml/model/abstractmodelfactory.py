from abc import ABC, abstractmethod

from myautoml.model.basemodel import BaseModel


class AbstractModelFactory(ABC):

    @abstractmethod
    def create_lightgbm(self) -> BaseModel:
        pass

    @abstractmethod
    def create_logreg(self) -> BaseModel:
        pass
