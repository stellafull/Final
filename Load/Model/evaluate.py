import pandas as pd
from joblib import load
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("match.csv")
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
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=30)
features_train_after, features_test_after, target_train_after, target_test_after = train_test_split(features_after, target, random_state=30)

xgb0 = load('output_model/xgb.joblib')
xgb1 = load('output_model/after/xgb.joblib')
lsvc = load('output_model/after/svc_linear.joblib')
psvc = load('output_model/after/svc_poly.joblib')
ssvc = load('output_model/after/svc_sigmoid.joblib')

# print("Xgboost: ", xgb.score(features_test, target_test), "auc: ", metrics.roc_auc_score(target_test, xgb.predict(features_test)))
# print("Linear:  ", lsvc.score(features_test, target_test), "auc: ", metrics.roc_auc_score(target_test, lsvc.predict(features_test)))
# print("Poly: ", psvc.score(features_test, target_test), "auc: ", metrics.roc_auc_score(target_test, psvc.predict(features_test)))
# print("Sigmoid: ", ssvc.score(features_test, target_test), "auc: ", metrics.roc_auc_score(target_test, ssvc.predict(features_test)))
# fig = plot_roc_curve(xgb0, features_test_after, target_test_after)
# fig = plot_roc_curve(xgb1, features_test,target_test)
# fig = plot_roc_curve(lsvc, features_test,target_test)
# fig = plot_roc_curve(psvc, features_test,target_test)
# fig = plot_roc_curve(ssvc, features_test,target_test)
# fig.figure_.suptitle("ROC curve comparison")
# plt.show()