from myautoml.preprocessor import *

class Loader(BaseLoader):

    def clean(self, df: DataFrame = None) -> DataFrame:
        pass

    def train_test_split(self, dataset, target_name):
        raise NotImplementedError('This is one not yet implemented. '
                                  'Use train_test_split from scikit-learn package instead.')