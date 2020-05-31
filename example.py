import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from myautoml import AutoML
from myautoml.data.datacontainer import DataContainer
from myautoml.model.modelregistryholder import ModelRegistryHolder

le = preprocessing.LabelEncoder()
if __name__ == '__main__':

    df = pd.read_csv('train.csv')
    df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df, df.Survived, test_size=0.2, random_state=42,)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


    ### encoding target step

    data = DataContainer(X_train=X_train,
                  X_test=X_test,
                  y_train=y_train,
                  y_test=y_test)

    #we can pass data, or path, or dataframe in future
    a = AutoML(data=data)
    a.fit()
