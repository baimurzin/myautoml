from __future__ import annotations

import multiprocessing
from typing import Optional, List

from myautoml.data.datacontainer import DataContainer
from myautoml.estimator import BaseEstimator
from myautoml.estimator import Classifier

import joblib


class AutoML:

    def __init__(self, estimator: BaseEstimator = None,
                 processes: int = 1,
                 verbose: bool = False,
                 model = None) -> None:
        self.model = model
        self.verbose = verbose
        self.processes = processes
        self._estimator = estimator
        self._automl_list: Optional[List] = []

    @property
    def estimator(self) -> BaseEstimator:
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: BaseEstimator) -> None:
        self._estimator = estimator

    def fit(self, data:DataContainer = None, **kwargs):
        self._automl_list = []
        detected_est = self.detect_estimator(data.y_train)
        self.estimator = detected_est

        if self.processes == 1:
            self.estimator.fit(data.X_train, data.y_train)
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


    def fit_predict(self):
        print('starting working')
        self._estimator.fit(dataset='')
        result = self._estimator.predict('')

    @staticmethod
    def detect_estimator(target) -> BaseEstimator:
        if (target.dtype == object) | (target.dtype == 'int'):
            return Classifier()
        else:
            pass

    @staticmethod
    def _fit(automl, **kwargs):
        return automl.fit(**kwargs)