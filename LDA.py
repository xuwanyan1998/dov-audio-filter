import sklearn.discriminant_analysis
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display

from IPython import display
from matplotlib import pyplot

train_df = pd.read_csv('data/featurized/featurized_train_data.csv')
test_df = pd.read_csv('data/featurized/featurized_test_data.csv')

X_df = train_df.loc[:,list(map(lambda c: c not in ['sample', 'trial', 'angle_pos', 'dov'], list(train_df.columns)))]
X = X_df.values
y_df = train_df.loc[:,"dov"]
# y_df = y_df.map(lambda x: int(x in [0,45,90,270,315]))
y = y_df.values
X[np.isnan(X)] = 0

X = X[:8000]
y = y[:8000]

lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
lda.fit(X, y)
X_new = lda.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
plt.show()