from __future__ import annotations

import multiprocessing
from typing import Optional, List

from myautoml.data.datacontainer import DataContainer

import joblib

from myautoml.model.abstractmodelfactory import AbstractModelFactory
from myautoml.model.classification.classifierfactory import ClassifierFactory
from myautoml.model.regression.regressorfactory import RegressorFactory


class AutoML:

    def __init__(self,
                 data: DataContainer = None,
                 processes: int = 1,
                 verbose: bool = False,
                 estimator: AbstractModelFactory = None,
                 model:Optional = None) -> None:
        self.model = model
        self.verbose = verbose
        self.data = data
        self.processes = processes
        self._automl_list: Optional[List] = [] #list of model objects to run in parallel
        if estimator is None:
            self.estimator = self.detect_estimator(data.y_train)
        else:
            self.estimator = estimator
        if model is None:
            self.model = self.set_default_model(estimator)
        else:
            self.model = model

    def _init_model(self, model:AbstractModelFactory):
        self.model = model


    def fit(self, **kwargs):
        self._automl_list = []
        data = self.data

        if self.processes == 1:
            self.model.fit(data.X_train, data.y_train)
        else:
            pass
            #do multiprocessing: create automl objects for each process
            #and handle data
        #     if self.processes == -1:
        #         self.n_processes = joblib.cpu_count()
        #     else:
        #         self.n_processes = self.processes
        #
        # processes = []
        # for i in range(self.n_processes):
        #     p = multiprocessing.Process(
        #         target=self._fit,
        #         kwargs=dict(
        #             automl=self._automl_list[i],
        #             kwargs=kwargs
        #         ),
        #     )
        #     processes.append(p)
        #     p.start()
        #
        # for p in processes:
        #     p.join()
        return self


    def fit_predict(self, data: DataContainer):
        print('starting working')
        self.model.fit(data.X_train, data.y_train)
        result = self.model.predict(data.X_test)
        return result

    @staticmethod
    def detect_estimator(target) -> AbstractModelFactory:
        if (target.dtype == object) | (target.dtype == 'int'):
            return ClassifierFactory()
        else:
            return RegressorFactory()

    @staticmethod
    def _fit(automl, **kwargs):
        return automl.fit(**kwargs)

    def set_default_model(self, estimator):
        return self.estimator.create_logreg()