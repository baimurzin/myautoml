from lightgbm import LGBMClassifier

from myautoml.model.basemodel import BaseModel


class LightGBM(BaseModel):

    def fit(self, x, y):
        super().fit(x, y)

    def predict(self, x):
        res = super().predict(x)
        # do smth special for this class
        return res

    def _init_model(self):
        self.model = LGBMClassifier(
                n_estimators=500, learning_rate=0.05,
                colsample_bytree=0.8, subsample=0.9, nthread=-1, seed=0)