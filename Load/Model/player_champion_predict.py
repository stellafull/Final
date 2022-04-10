from joblib import dump, load
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split, KFold
import pandas as pd

df = pd.read_csv("match.csv")

# print('Dimension of dataset= ', df.shape)

# skf = StratifiedKFold(n_splits=10, shuffle = True, random_state=0)     # produce same seed every time
features = df.drop('result', axis=1)
target = df['result']
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=12)
