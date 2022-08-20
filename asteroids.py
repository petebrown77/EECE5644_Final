import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import pandas as pd

from scipy.stats import multivariate_normal as mvn
from scipy.optimize import minimize as optmin
from scipy.stats import reciprocal

from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from scikeras.wrappers import KerasClassifier, KerasRegressor

np.random.seed(69+420)


asteroids = pd.read_csv("asteroids.csv", delimiter=",")

#drop the data that's obviously never-useful
asteroids = asteroids.drop(columns=['name', 'orbiting_body', 'sentry_object'])
asteroids['hazardous'] = asteroids['hazardous']*1

x = asteroids.drop(columns=['id', 'hazardous'])
y = asteroids['hazardous']


X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)


# sns.pairplot(data=asteroids[['absolute_magnitude', 'est_diameter_max', 'relative_velocity', 'miss_distance', 'hazardous']], hue='hazardous', plot_kws={'alpha':0.1})

# ax = sns.jointplot(data=asteroids, x='est_diameter_min', y='absolute_magnitude', hue='hazardous', alpha=0.1)

# ax.ax_joint.set_xscale('log')

# plt.tight_layout()
# plt.show()


# >>> asteroids.corr()
#                           id  est_diameter_min  est_diameter_max  relative_velocity  miss_distance  absolute_magnitude  hazardous
# id                  1.000000         -0.148322         -0.148322          -0.059176      -0.056510            0.277258  -0.123443
# est_diameter_min   -0.148322          1.000000          1.000000           0.221553       0.142241           -0.560188   0.183363
# est_diameter_max   -0.148322          1.000000          1.000000           0.221553       0.142241           -0.560188   0.183363
# relative_velocity  -0.059176          0.221553          0.221553           1.000000       0.327169           -0.353863   0.191185
# miss_distance      -0.056510          0.142241          0.142241           0.327169       1.000000           -0.264168   0.042302
# absolute_magnitude  0.277258         -0.560188         -0.560188          -0.353863      -0.264168            1.000000  -0.365267
# hazardous          -0.123443          0.183363          0.183363           0.191185       0.042302           -0.365267   1.000000



def build_model(nNeurons=30, lr = .015, input_shape=[5]):
	model = Sequential()
	model.add(keras.layers.InputLayer(input_shape=[5]))
	model.add(keras.layers.Dense(nNeurons, activation='tanh'))
	model.add(keras.layers.Dense(1, activation='sigmoid'))
	optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True, decay=1e-5)
	model.compile(loss= tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer,  metrics=[tf.keras.metrics.FalsePositives()])
	return model


# train model on good params
model = build_model(nNeurons=30)
keras_clf = KerasClassifier(model=model)
keras_clf.fit(X_train, y_train, epochs=45,
	validation_data=(X_test, y_test))
