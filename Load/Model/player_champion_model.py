import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, __all__
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC, LinearSVC
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn import tree
from sklearn.metrics import RocCurveDisplay


def svm_select(training_features , training_target):
    lsvc = Pipeline([('scaler', StandardScaler()), ('svm', LinearSVC(C=0.01, penalty="l1", dual=False))]).fit(training_features, training_target)
    selector = SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False)).fit(training_features, training_target)
    importance = lsvc.named_steps['svm'].coef_
    # dump(lsvc, 'output_model/svm.joblib')
    # training_features_new = model.transform(training_features)
    # score = lsvc.score(testing_features, testing_target)
    # print("Shape after feature selecting", training_features_new.shape)
    # print(pipe.score(testing_features, testing_target))
    return importance


# def LR_L1(training_features, testing_features, training_target, testing_target):
#     pipe = Pipeline([('scaler', StandardScaler()), ('LR', LogisticRegression())])
#     pipe.fit(training_features, training_target)
#     dump(pipe, 'output_model/LR_L1.joblib')
#     score = pipe.score(testing_features, testing_target)
#     return score

# def forest_train(training_features, testing_features, training_target, testing_target):
#     clf = RandomForestClassifier(max_depth=1, random_state=0)
#     clf.fit(training_features, training_target)
#     dump(clf, 'output_model/forest.joblib')
#     score = clf.score(testing_features, testing_target)
#     return score


# def tree_train(training_features, testing_features, training_target, testing_target):
#     clf = tree.DecisionTreeClassifier().fit(training_features, training_target)
#     clf.fit(training_features, training_target)
#     dump(clf, 'output_model/tree.joblib')
#     score = clf.score(testing_features, testing_target)
#     return score


# def lasso_selection(features, target):
#     X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=42)
#     lasso = Pipeline([('scaler', StandardScaler()), ('model', Lasso())])
#     search = GridSearchCV(lasso,
#                           {'model__alpha': np.arange(0.1, 10, 0.1)},
#                           cv=5, scoring="neg_mean_squared_error", verbose=3
#                           )
#     search.fit(X_train, y_train)
#     coefficients = search.best_estimator_.named_steps['model'].coef_
#     importance = np.abs(coefficients)
#     features_select = np.array(features)[importance > 0]
#     return features_select


# def log_sleect(features, target):
#     selector = SelectFromModel(estimator=LogisticRegression()).fit(features, target)
#     return selector.get_support()


# def log_l1_train(features, target, skf):
#     log = Pipeline([('scaler', StandardScaler()), ('log', LogisticRegression(penalty='l1'))])
#     param_grid = [
#         {
#             "log__degree": [2, 3],
#             "log__C": [0.1, 0.25, 0.5, 1, 1.5, 2, 4, 8, 10, 16, 25, 32, 64, 100],
#             "log__gamma": [0.5, 1, 2, 3, 4]
#         }
#     ]


# def log_l2_train(features, target, skf):
# def log_elasticnet_train(features, target, skf):


def xgboost_train(features, target, skf):
    xgb = Pipeline([('scaler', StandardScaler()), ('xgboost', GradientBoostingClassifier())])
    param_grid = [
        {
            "xgboost__n_estimators": [100, 125, 150, 175, 200],
            "xgboost__min_samples_split": [2, 4, 6, 8],
            "xgboost__max_depth": range(2, 6)
        }
    ]
    grid = GridSearchCV(xgb, param_grid, refit=True, verbose=2, cv=skf)
    grid.fit(features, target)
    print(grid.best_estimator_)
    return grid.best_estimator_


def svc_linear_train(features, target, skf):
    lsvc = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='linear'))])
    param_grid = [
        {
            # "svm__kernel": ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            "svm__degree": [2, 3],
            "svm__C": [0.01, 0.1, 0.25, 0.5, 1, 1.5, 2, 4, 8, 10, 16, 25, 32, 64, 100],
            "svm__gamma": [0.01, 0.1, 0.25, 0.5, 1, 2, 3, 4]
        }
    ]
    grid = GridSearchCV(lsvc, param_grid, refit=True, verbose=2, cv=skf)
    # model = SelectFromModel(lsvc, prefit=True)
    # dump(lsvc, 'output_model/lsvc_l1.joblib')
    # scores = cross_val_score(grid, features, target, cv=skf, scoring="recall")
    grid.fit(features, target)
    print(grid.best_estimator_)
    # print(pipe.score(testing_features, testing_target))
    # return scores.mean()
    return grid.best_estimator_


def svc_poly_train(features, target, skf):
    lsvc = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='poly'))])
    param_grid = [
        {
            "svm__degree": [2, 3],
            "svm__C": [0.01, 0.1, 0.25, 0.5, 1, 1.5, 2, 4, 8, 10, 16, 25, 32, 64, 100],
            "svm__gamma": [0.01, 0.1, 0.25, 0.5, 1, 2, 3, 4]
        }
    ]
    grid = GridSearchCV(lsvc, param_grid, refit=True, verbose=2, cv=skf)
    grid.fit(features, target)
    print(grid.best_estimator_)
    return grid.best_estimator_


def svc_rbf_train(features, target, skf):
    lsvc = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf'))])
    param_grid = [
        {
            "svm__degree": [2, 3],
            "svm__C": [0.01, 0.1, 0.25, 0.5, 1, 1.5, 2, 4, 8, 10, 16, 25, 32, 64, 100],
            "svm__gamma": [0.01, 0.1, 0.25, 0.5, 1, 2, 3, 4]
        }
    ]
    grid = GridSearchCV(lsvc, param_grid, refit=True, verbose=2, cv=skf)
    grid.fit(features, target)
    print(grid.best_estimator_)
    return grid.best_estimator_


def svc_sigmoid_train(features, target, skf):
    lsvc = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='sigmoid'))])
    param_grid = [
        {
            "svm__degree": [2, 3],
            "svm__C": [0.01, 0.1, 0.25, 0.5, 1, 1.5, 2, 4, 8, 10, 16, 25, 32, 64, 100],
            "svm__gamma": [0.01, 0.1, 0.25, 0.5, 1, 2, 3, 4]
        }
    ]
    grid = GridSearchCV(lsvc, param_grid, refit=True, verbose=2, cv=skf)
    grid.fit(features, target)
    print(grid.best_estimator_)
    return grid.best_estimator_


def svc_precomputed_train(features, target, skf):
    lsvc = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='precomputed'))])
    param_grid = [
        {
            "svm__degree": [2, 3],
            "svm__C": [0.01, 0.1, 0.25, 0.5, 1, 1.5, 2, 4, 8, 10, 16, 25, 32, 64, 100],
            "svm__gamma": [0.01, 0.1, 0.25, 0.5, 1, 2, 3, 4]
        }
    ]
    grid = GridSearchCV(lsvc, param_grid, refit=True, verbose=2, cv=skf)
    grid.fit(features, target)
    print(grid.best_estimator_)
    return grid.best_estimator_

# feature_file = csv.writer(open("feature.csv", "w", newline=""))
df = pd.read_csv("match.csv")
importance_file = csv.writer(open("output_model/importance.csv", "w",newline=""), delimiter=',')
# print('Dimension of dataset= ', df.shape)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)  # produce same seed every time
features = df.drop('result', axis=1)
target = df['result']
features_after = df[
    ["blue_player1_KDA", "blue_player1_GPM", "blue_player1_DPM", "blue_champion1_DPM", "blue_champion1_GPM",
     "blue_player2_KDA", "blue_player2_GPM", "blue_player2_DPM", "blue_champion2_DPM", "blue_champion2_GPM",
     "blue_player3_KDA", "blue_player3_GPM", "blue_player3_DPM", "blue_champion3_DPM", "blue_champion3_GPM",
     "blue_player4_KDA", "blue_player4_GPM", "blue_player4_DPM", "blue_champion4_DPM", "blue_champion4_GPM",
     "blue_player5_KDA", "blue_player5_GPM", "blue_player5_DPM", "blue_champion5_DPM", "blue_champion5_GPM",
     "red_player1_KDA", "red_player1_GPM", "red_player1_DPM", "red_champion1_DPM", "red_champion1_GPM",
     "red_player2_KDA", "red_player2_GPM", "red_player2_DPM", "red_champion2_DPM", "red_champion2_GPM",
     "red_player3_KDA", "red_player3_GPM", "red_player3_DPM", "red_champion3_DPM", "red_champion3_GPM",
     "red_player4_KDA", "red_player4_GPM", "red_player4_DPM", "red_champion4_DPM", "red_champion4_GPM",
     "red_player5_KDA", "red_player5_GPM", "red_player5_DPM", "red_champion5_DPM", "red_champion5_GPM"]]
# features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=30)
features_train, features_test, target_train, target_test = train_test_split(features_after, target, random_state=30)
features_importance = []

# feature selection
for train_index, test_index in skf.split(features, target):
    training_features, testing_features = features.iloc[train_index, :], features.iloc[test_index, :]
    training_target, testing_target = target[train_index], target[test_index]
    features_importance.append(svm_select(training_features, training_target))
    # print(features_importance)
    # importance_file.writerow(features_importance)
#     feature_name = features.columns[svm_select(training_features, training_target)]
#     feature_file.writerow(feature_name)
#     print(feature_name)
df = pd.DataFrame(np.array(features_importance).reshape(10, 1*110))
df.to_csv("output_model/importance.csv")



# print(sum(svc_train(features, target, skf))/10)
# lasso_selection(features, target)


# print(svc_train(features_train, target_train, skf))
# print(log_sleect(features_train, target_train))
# feature_name = features.columns[log_sleect(features_train, target_train)]
# print(feature_name)
xgboost_best = xgboost_train(features_train, target_train,skf)
svc_precomputed_best = svc_precomputed_train(features_train, target_train,skf)
svc_linear_best = svc_linear_train(features_train, target_train,skf)
svc_rbf_best = svc_rbf_train(features_train, target_train,skf)
svc_poly_best = svc_poly_train(features_train, target_train,skf)
svc_sigmoid_best = svc_sigmoid_train(features_train, target_train,skf)
print(xgboost_best)
print(svc_sigmoid_best)
print(svc_poly_best)
print(svc_rbf_best)
print(svc_linear_best)


# for train_index, test_index in skf.split(features_after, target):
#     training_features, testing_features = features_after.iloc[train_index, :], features_after.iloc[test_index, :]
#     training_target, testing_target = target[train_index], target[test_index]
#     XGboost_score.append(XGboost_train(training_features, testing_features, training_target, testing_target))
# svm_score.append(svm_train(training_features, testing_features, training_target, testing_target))
# LR_L1_score.append(LR_L1(training_features, testing_features, training_target, testing_target))
# tree_score.append(tree_train(training_features, testing_features, training_target, testing_target))
# forest_score.append(forest_train(training_features, testing_features, training_target, testing_target))
# model.fit(X_train, y_train)
# pred_values = model.predict(X_test)
#
# acc = accuracy_score(pred_values, y_test)
# acc_score.append(acc)