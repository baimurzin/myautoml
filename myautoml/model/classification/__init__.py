from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

class_models = {
    'LightGBM':  LGBMClassifier(
                n_estimators=500, learning_rate=0.05,
                colsample_bytree=0.8, subsample=0.9, nthread=-1, seed=0),
    'DecisionTree': DecisionTreeClassifier(
                criterion='gini', splitter='best', max_depth=None,
                min_samples_split=2, min_samples_leaf=1,
                min_weight_fraction_leaf=0.0, max_features=None,
                random_state=0, max_leaf_nodes=None, class_weight=None,
                presort=False),
    'AdaBoost': AdaBoostClassifier(
                base_estimator=None, n_estimators=400, learning_rate=.05,
                algorithm='SAMME.R', random_state=0),
    'LogisticRegression': LogisticRegression(
                penalty='l2', dual=False, tol=0.0001, C=1.0,
                fit_intercept=True, intercept_scaling=1, class_weight=None,
                random_state=0, solver='lbfgs', max_iter=100,
                multi_class='ovr', verbose=0, warm_start=False, n_jobs=-1),
    'RandomForest': RandomForestClassifier(
                n_estimators=400, max_depth=10, max_features='sqrt',
                bootstrap=True, n_jobs=-1, random_state=0)

}