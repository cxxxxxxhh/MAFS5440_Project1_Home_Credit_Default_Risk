import pandas as pd
import numpy as np
import optuna
import operator
#from xgboost import XGBRegressor

from pandas import read_parquet
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    make_scorer, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier


final_data=pd.read_parquet("/home/cheam/fin_project1/fintech_pro1_data/train_data_flit.parquet")


class EarlyStoppingCallback(object):
    """Early stopping callback for Optuna."""

    def __init__(self, early_stopping_rounds: int, direction: str = "minimize") -> None:
        self.early_stopping_rounds = early_stopping_rounds

        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Do early stopping."""
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()


# optuna 优化函数
def objective(trial): #先定义很多超参数的范围，然后利用这些参数训练模型，得到test error最小的，假设我们是不知道test data的吧，只能输出cv的test error

    learning_rate=trial.suggest_float("learning_rate", 0, 0.2)
    #max_depth=trial.suggest_int("max_depth", 3, 16),
    max_depth = trial.suggest_int("max_depth", 3, 16)
    #print(type(max_depth))
    num_leaves=trial.suggest_int("num_leaves", 8, 2**max_depth)
    min_data_in_leaf= trial.suggest_int("min_data_in_leaf", 5, 300)
    feature_fraction= trial.suggest_float("feature_fraction", 0.5, 1)
    bagging_fraction= trial.suggest_float("bagging_fraction", 0.5, 1)
    bagging_freq= trial.suggest_int("bagging_freq", 1,10)
    lambda_l1= trial.suggest_float("lambda_l1", 0.01, 100)
    lambda_l2= trial.suggest_float("lambda_l2", 0.01, 100)
    min_gain_to_split= trial.suggest_int("min_gain_to_split", 0, 15)
    n_estimators= trial.suggest_int("n_estimators", 1,2000)
    scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),  
    'recall': make_scorer(recall_score),  
    'f1': make_scorer(f1_score)  
}

    try:

        classifier = lgb.LGBMClassifier(random_state=42,learning_rate = learning_rate, max_depth = max_depth, 
                                        num_leaves = num_leaves, min_data_in_leaf = min_data_in_leaf,feature_fraction = feature_fraction,
                                        bagging_fraction = bagging_fraction, bagging_freq = bagging_freq, lambda_l1 = lambda_l1,
                                        lambda_l2 = lambda_l2, min_gain_to_split = min_gain_to_split, n_estimators = n_estimators)

        f1_scores = cross_val_score(classifier, final_data.drop('TARGET', axis =1 ), final_data['TARGET'], cv=5, scoring=scoring['f1'])
        return f1_scores.mean()
    except:
    #     #print(e)
        return -10000
    
    


direction = "maximize"
# 创建study
study = optuna.create_study(
    storage="sqlite:///parameter_tunning_project_1.db",
    study_name="tunning_lgbm_flit",
    direction=direction,
    load_if_exists=True,
)

early_stopping = EarlyStoppingCallback(10, direction=direction)


# 开始优化
study.optimize(objective, n_jobs=1, callbacks=[early_stopping], timeout=3600*1)


# 输出结果
print(f"Best trial number: {study.best_trial.number}")
print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")