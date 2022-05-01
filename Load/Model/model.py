import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import dump

from Load.Model.draw import draw_cv_roc_curve, draw_cv_pr_curve

df = pd.read_csv("match.csv")
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
features_train_after, features_test_after, target_train_after, target_test_after = train_test_split(features_after,
                                                                                                    target,
                                                                                                    random_state=30)
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=30)


def xgboost_model(features, target, skf):
    xgb = Pipeline(
        [('scaler', StandardScaler()), ('xgboost', GradientBoostingClassifier(max_depth=2, n_estimators=125))])
    scores = cross_val_score(xgb, features, target, cv=skf, scoring='accuracy')
    draw_cv_roc_curve(xgb, cv=skf, X=features, y=target, title='Cross Validated ROC')
    draw_cv_pr_curve(xgb, cv=skf, X=features, y=target, title='Cross Validated ROC')
    xgb.fit(features, target)
    dump(xgb, "output_model/xgb.joblib")
    return scores.mean()


def xgboost_model_nan(features, target, skf):
    xgb = Pipeline(
        [('scaler', StandardScaler()), ('xgboost', GradientBoostingClassifier(max_depth=2, n_estimators=125))])
    scores = cross_val_score(xgb, features, target, cv=skf, scoring='accuracy')
    draw_cv_roc_curve(xgb, cv=skf, X=features, y=target, title='Cross Validated ROC')
    draw_cv_pr_curve(xgb, cv=skf, X=features, y=target, title='Cross Validated ROC')
    xgb.fit(features, target)
    dump(xgb, "output_model/after/xgb.joblib")
    return scores.mean()


def svm_linear_model(features, target, skf):
    lsvc = Pipeline(
        [('scaler', StandardScaler()), ('svm', SVC(kernel='linear', C=100, degree=2, gamma=0.5, probability=True))])
    scores = cross_val_score(lsvc, features, target, cv=skf, scoring='accuracy')
    draw_cv_roc_curve(lsvc, cv=skf, X=features, y=target, title='Cross Validated ROC')
    draw_cv_pr_curve(lsvc, cv=skf, X=features, y=target, title='Cross Validated ROC')
    lsvc.fit(features, target)
    dump(lsvc, "output_model/after/svc_linear.joblib")
    return scores.mean()


def svc_poly_model(features, target, skf):
    psvc = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='poly', C=0.1, gamma=0.5, probability=True))])
    scores = cross_val_score(psvc, features, target, cv=skf, scoring='accuracy')
    draw_cv_roc_curve(psvc, cv=skf, X=features, y=target, title='Cross Validated ROC')
    draw_cv_pr_curve(psvc, cv=skf, X=features, y=target, title='Cross Validated ROC')
    psvc.fit(features, target)
    dump(psvc, "output_model/after/svc_poly.joblib")
    return scores.mean()


def svm_rbf_model(features, target, skf):
    rsvc = Pipeline(
        [('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=0.1, degree=2, gamma=0.01, probability=True))])
    scores = cross_val_score(rsvc, features, target, cv=skf, scoring='accuracy')
    draw_cv_roc_curve(rsvc, cv=skf, X=features, y=target, title='Cross Validated ROC')
    draw_cv_pr_curve(rsvc, cv=skf, X=features, y=target, title='Cross Validated ROC')
    rsvc.fit(features, target)
    dump(rsvc, "output_model/after/svc_rbf.joblib")
    return scores.mean()


def svm_sigmoid_train(features, target, skf):
    ssvc = Pipeline(
        [('scaler', StandardScaler()), ('svm', SVC(kernel='sigmoid', C=0.1, degree=2, gamma=1, probability=True))])
    scores = cross_val_score(ssvc, features, target, cv=skf, scoring='accuracy')
    draw_cv_roc_curve(ssvc, cv=skf, X=features, y=target, title='Cross Validated ROC')
    draw_cv_pr_curve(ssvc, cv=skf, X=features, y=target, title='Cross Validated ROC')
    ssvc.fit(features, target)
    dump(ssvc, "output_model/after/svc_sigmoid.joblib")
    return scores.mean()


print("xgboost1: ", xgboost_model(features_train_after, target_train_after, skf))
print("xgboost2: ", xgboost_model_nan(features_train, target_train, skf))
print("svm_linear: ", svm_linear_model(features_train, target_train, skf))
print("svm_poly: ", svc_poly_model(features_train, target_train, skf))
print("svm_rbf: ", svm_rbf_model(features_train, target_train, skf))
print("svm_sigmoid: ", svm_sigmoid_train(features_train, target_train, skf))
# xgboost_model(features_train, target_train, skf)
