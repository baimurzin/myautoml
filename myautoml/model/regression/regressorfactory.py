from abc import abstractmethod

from pandas import DataFrame, Series

from myautoml.model.abstractmodelfactory import AbstractModelFactory
from myautoml.model.basemodel import BaseModel


class RegressorFactory(AbstractModelFactory):

    def create_lightgbm(self) -> BaseModel:
        pass

    def create_logreg(self) -> BaseModel:
        pass