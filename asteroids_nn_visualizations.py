import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import pandas as pd

from scipy.stats import multivariate_normal as mvn
from scipy.optimize import minimize as optmin
from scipy.stats import reciprocal

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from focal_loss import BinaryFocalLoss
from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
np.random.seed(69+420)

# importing data
asteroids = pd.read_csv("asteroids.csv", delimiter=",")

#drop the data that's obviously never-useful
asteroids = asteroids.drop(columns=['name', 'orbiting_body', 'sentry_object'])
asteroids['hazardous'] = asteroids['hazardous']*1

# what are our correlations?
sns.heatmap(asteroids.corr(), annot=True)
plt.show()

# form data and labels
x = asteroids.drop(columns=['id', 'hazardous'])
y = asteroids['hazardous']

neg, pos = np.bincount(asteroids['hazardous'])
total = neg + pos
print('Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

# splits
X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)

# run through scaler to improve performance on some models
scaler = StandardScaler()

# keep everything as a dataframe, resisting the urge of the scaler
cols = X_train.columns
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)
X_test  = pd.DataFrame(scaler.transform(X_test), columns=cols)


# generates data about raw distribution, takes a while
'''
sns.pairplot(data=asteroids[['absolute_magnitude', 'est_diameter_max', 'relative_velocity', 'miss_distance', 'hazardous']], hue='hazardous', plot_kws={'alpha':0.1})

ax = sns.jointplot(data=asteroids, x='est_diameter_min', y='absolute_magnitude', hue='hazardous', alpha=0.1)

ax.ax_joint.set_xscale('log')

plt.tight_layout()
plt.show()
'''

'''
>>> asteroids.corr()
                          id  est_diameter_min  est_diameter_max  relative_velocity  miss_distance  absolute_magnitude  hazardous
id                  1.000000         -0.148322         -0.148322          -0.059176      -0.056510            0.277258  -0.123443
est_diameter_min   -0.148322          1.000000          1.000000           0.221553       0.142241           -0.560188   0.183363
est_diameter_max   -0.148322          1.000000          1.000000           0.221553       0.142241           -0.560188   0.183363
relative_velocity  -0.059176          0.221553          0.221553           1.000000       0.327169           -0.353863   0.191185
miss_distance      -0.056510          0.142241          0.142241           0.327169       1.000000           -0.264168   0.042302
absolute_magnitude  0.277258         -0.560188         -0.560188          -0.353863      -0.264168            1.000000  -0.365267
hazardous          -0.123443          0.183363          0.183363           0.191185       0.042302           -0.365267   1.000000
'''

# does what it says on the box
METRICS = [
	keras.metrics.TruePositives(name='tp'),
	keras.metrics.FalsePositives(name='fp'),
	keras.metrics.TrueNegatives(name='tn'),
	keras.metrics.FalseNegatives(name='fn'),
	keras.metrics.BinaryAccuracy(name='accuracy'),
	keras.metrics.Precision(name='precision'),
	keras.metrics.Recall(name='recall'),
	keras.metrics.AUC(name='auc'),
	keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

# good for wrapping our model in sklearn API, quick builds
def build_model(nNeurons=30, lr = .035, input_shape=[5], gamma=2):
	model = Sequential()
	model.add(keras.layers.InputLayer(input_shape=[5]))
	model.add(keras.layers.Dense(nNeurons, activation='tanh'))
	model.add(keras.layers.Dense(nNeurons, activation='tanh'))
	model.add(keras.layers.Dense(nNeurons, activation='tanh'))
	model.add(keras.layers.Dense(1, activation='sigmoid'))
	optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True, decay=1e-5)
	model.compile(loss=BinaryFocalLoss(gamma=gamma), optimizer=optimizer, metrics=METRICS)
	return model

#set up callbacks and misc. model variables
tensorboard = TensorBoard(log_dir="logs/{}".format((time())))
callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5), tensorboard]

class_weight = {
		0 : 1,
		1 : 10}

# train model
model = build_model(nNeurons=20, gamma=1)
keras_clf = KerasClassifier(model=model)

print(model.summary())

keras_clf.fit(X_train, y_train, epochs=45, validation_data=(X_test, y_test),  callbacks=[callback], class_weight = class_weight)

#UNTUNED MODEL EVALUATION AND VISUALIZATIONS
preds = keras_clf.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, preds)
plt.show()

asteroids_test = X_test
asteroids_test['hazardous'] = y_test.values
asteroids_test['preds'] = preds

#gross, but short! Gives FP TP FN TN encodings
conf = ((preds==1) & (y_test.values==1))*1 + ((preds==0) & (y_test.values==1))*2 + ((preds==1) & (y_test.values==0))*3 + ((preds==0) & (y_test.values==0))*4
asteroids_test['conf'] = conf
asteroids_test['conf'] = asteroids_test['conf'].replace({1:'TP', 2:'FN', 3:'FP', 4:'TN'})

palette = {'TP': 'tab:green',
			'FP': 'tab:orange',
			'FN': 'tab:red',
			'TN': 'tab:blue'}

# SCATTERPLOTS
#miss distance, absolute magnitude
sns.jointplot(data=asteroids_test, x='miss_distance', y='absolute_magnitude', hue='conf',  palette=palette,  alpha=.2)
sns.jointplot(data=asteroids_test.where(asteroids_test['hazardous']==1), x='miss_distance', y='absolute_magnitude', hue='conf',  palette=palette, alpha=.2)
sns.jointplot(data=asteroids_test.where(asteroids_test['hazardous']==0), x='miss_distance', y='absolute_magnitude', hue='conf',   palette=palette, alpha=.2)

#relative velocity, absolute magnitude
sns.jointplot(data=asteroids_test, x='relative_velocity', y='absolute_magnitude', hue='conf', palette=palette, alpha=.2)
sns.jointplot(data=asteroids_test.where(asteroids_test['hazardous']==1), x='relative_velocity', y='absolute_magnitude', hue='conf', palette=palette, alpha=.2)
sns.jointplot(data=asteroids_test.where(asteroids_test['hazardous']==0), x='relative_velocity', y='absolute_magnitude', hue='conf', palette=palette,  alpha=.2)

#miss distance, relative velocity
sns.jointplot(data=asteroids_test, x='relative_velocity', y='miss_distance', hue='conf', palette=palette, alpha=.2)
sns.jointplot(data=asteroids_test.where(asteroids_test['hazardous']==1), x='relative_velocity', y='miss_distance', hue='conf', palette=palette, alpha=.2)
sns.jointplot(data=asteroids_test.where(asteroids_test['hazardous']==0), x='relative_velocity', y='miss_distance', hue='conf', palette=palette,  alpha=.2)

#lower triangle plot
sns.pairplot(asteroids_test[['miss_distance', 'conf', 'relative_velocity', 'est_diameter_max', 'absolute_magnitude', 'hazardous']], hue='conf', palette=palette, corner = True)


# KDE PLOTS
# miss distance, absolute magnitude
sns.kdeplot(data=asteroids_test, x='miss_distance', y='absolute_magnitude', hue='conf',  palette=palette)

#relative velocity, absolute magnitude
sns.kdeplot(data=asteroids_test, x='relative_velocity', y='absolute_magnitude', hue='conf', palette=palette, )

#miss distance, relative velocity
sns.kdeplot(data=asteroids_test, x='relative_velocity', y='miss_distance', hue='conf', palette=palette)

#lower triangle plot
sns.pairplot(asteroids_test[['miss_distance', 'conf', 'relative_velocity', 'est_diameter_max', 'absolute_magnitude']], hue='conf', palette=palette, corner = True, kind="kde")

# RAW DATA DIST
# lower triangle plot
sns.pairplot(asteroids_test[['miss_distance', 'hazardous', 'relative_velocity', 'est_diameter_max', 'absolute_magnitude']], hue='hazardous', corner = True, kind="kde")

plt.tight_layout()
plt.show()

