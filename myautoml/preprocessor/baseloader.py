from abc import ABC, abstractmethod

from pandas import DataFrame

class BaseLoader(ABC):

    def __init__(self,
                 separator=',',
                 header=False,
                 save_path='automl_results',
                 verbose=False) -> None:
        self.save_path = save_path
        self.header = header
        self.separator = separator
        self.verbose = verbose
        super().__init__()

    @abstractmethod
    def train_test_split(self, dataset, target_name):
        pass

    @abstractmethod
    def clean(self, df : DataFrame = None) -> DataFrame:
        pass
