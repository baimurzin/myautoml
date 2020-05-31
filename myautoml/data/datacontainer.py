import pandas as pd
import numpy as np
from category_encoders import JamesSteinEncoder, OneHotEncoder, HelmertEncoder


class DataContainer(object):

    encoders = {
        'OneHotEncoder': OneHotEncoder,
        'HelmertEncoder': HelmertEncoder
    }

    def __init__(self,
                 X_train=None,
                 y_train=None,
                 X_test=None,
                 y_test=None,
                 cat_features=None,
                 preprocessing=True
                 ) -> None:
        self.target_encoder_name = JamesSteinEncoder
        if y_train is None:
            raise Exception('no target provided')
        else:
            self.y_train = y_train
        self.X_train_origin = self.dataframe_format(X_train)
        if X_test is not None:
            self.X_test_origin = self.dataframe_format(X_test)
        if y_test is not None:
            self.y_test = y_test
        if cat_features is None:
            self.cat_features = self.detect_cat_features(self.X_train_origin)
        else:
            self.cat_features = list(cat_features)

        if preprocessing:
            self.X_train, self.X_test = self.preproc_data(self.X_train_origin,
                                                          self.X_test_origin,
                                                          cat_features=self.cat_features,
                                                          cat_encoder_name='HelmertEncoder')
        else:
            self.X_train, self.X_test = X_train, X_test
        print('finished')



    @staticmethod
    def dataframe_format(X_train):
        tmp = pd.DataFrame(X_train)
        if tmp is None or tmp.empty:
            raise Exception("bad format for data")
        return tmp

    @staticmethod
    def detect_cat_features(data):
        cat_features = data.columns[(data.nunique(dropna=False) < len(data) // 100) & \
                                    (data.nunique(dropna=False) > 2)]
        return cat_features


    def preproc_data(self, X_train=None, X_test=None, cat_features=None, cat_encoder_name=None):

        df_train = X_train.copy()
        df_train['test'] = 0

        if X_test is not None:
            df_test = X_test.copy()
            df_test['test'] = 1
            data = df_train.append(df_test, sort=False).reset_index(drop=True)  # concat
        else:
            data = df_train

        # object & num features
        object_features = list(data.columns[(data.dtypes == 'object') | (data.dtypes == 'category')])
        num_features = list(set(data.columns) - set(cat_features) - set(object_features) - {'test'})
        encoded_features_names = list(set(object_features + list(cat_features)))
        self.encoded_features_names = encoded_features_names
        # LabelEncoded Binary Features
        for feature in data.columns:
            if (feature is not 'test') and (data[feature].nunique(dropna=False) < 3):
                data[feature] = data[feature].astype('category').cat.codes

        if encoded_features_names:
            encoder = self.encoders[cat_encoder_name](drop_invariant=True)
            if cat_encoder_name == 'HashingEncoder':
                encoder = cat_encoder_name(n_components=int(np.log(len(data))*100),
                                                        drop_invariant=True)
            data_encoded = encoder.fit_transform(data[encoded_features_names])
            data_encoded = data_encoded.add_prefix(cat_encoder_name + '_')


            if self.target_encoder_name is not None:
                data = pd.concat([
                    data.reset_index(drop=True),
                    data_encoded.reset_index(drop=True)],
                    axis=1,)
            else:
                data = pd.concat([
                    data.drop(columns=encodet_features_names).reset_index(drop=True),
                    data_encoded.reset_index(drop=True)],
                    axis=1,)
        data = self.clean_nans(data, cols=num_features)

        X_train = data.query('test == 0').drop(['test'], axis=1)
        X_test = data.query('test == 1').drop(['test'], axis=1)
        return X_train, X_test

    @staticmethod
    def clean_nans(data, cols=None):
        if cols is not None:
            nan_columns = list(data[cols].columns[data[cols].isnull().sum() > 0])
            if nan_columns:
                for nan_column in nan_columns:
                    data[nan_column + 'isNAN'] = pd.isna(data[nan_column]).astype('uint8')
                data.fillna(data.median(), inplace=True)
        return data