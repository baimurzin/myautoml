from myautoml.model.abstractmodelfactory import AbstractModelFactory
from myautoml.model.basemodel import BaseModel
from myautoml.model.classification.lightgbm import LightGBM
from myautoml.model.classification.logreg import LogReg


class ClassifierFactory(AbstractModelFactory):

    def create_lightgbm(self) -> BaseModel:
        model = LightGBM()
        model._init_model() ##init some params
        return model

    def create_logreg(self) -> BaseModel:
        model = LogReg()
        model._init_model()
        return model