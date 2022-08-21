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

# form data and labels
x = asteroids.drop(columns=['id', 'hazardous'])
y = asteroids['hazardous']

# splits
X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)

# run through scaler to improve performance on some models
scaler = StandardScaler()

# keep everything as a dataframe, resisting the urge of the scaler
cols = X_train.columns
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)
X_test  = pd.DataFrame(scaler.transform(X_test), columns=cols)



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


tensorboard = TensorBoard(log_dir="logs/{}".format((time())))
callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5), tensorboard]

class_weight = {
		0 : 1,
		1 : 10}


#working with the keras tuner system
def keras_build_model(hp):
	model = keras.Sequential()
	model.add(keras.layers.InputLayer(input_shape=[5]))
	# Tune the number of layers.
	for i in range(hp.Int("num_layers", 2, 4)):
		model.add(keras.layers.Dense(hp.Int("units", min_value=15, max_value=30, step=5), activation='tanh'))
	model.add(keras.layers.Dense(1, activation='sigmoid'))
	lr = hp.Float("lr", min_value=1e-4, max_value=0.03, sampling="log")
	gamma = hp.Float("gamma", min_value=0.05, max_value = 5, sampling="log")

	optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True, decay=1e-5)
	model.compile(loss=BinaryFocalLoss(gamma=gamma), optimizer=optimizer, metrics=METRICS)
	return model

import keras_tuner

# model = keras_build_model(keras_tuner.HyperParameters())

tuner = keras_tuner.RandomSearch(
	hypermodel=keras_build_model,
	objective=keras_tuner.Objective("val_recall", direction="max"),
	max_trials=3,
	executions_per_trial=5,
	overwrite=True,
	directory="asteroids_nn_tuning",
	project_name="asteroids",
)

tuner.search(X_train,
			y_train,
			epochs=10,
			validation_data=(X_test, y_test),
			callbacks=[callback],
			class_weight = class_weight)

best_hp = tuner.get_best_hyperparameters()[0]
best_hp.values
# {'num_layers': 3, 'units': 30, 'lr': 0.009263303924328512, 'gamma': 0.11835983615085716}

# summary = tuner.results_summary(num_trials=3)

# Results summary
# Results in asteroids_nn_tuning/asteroids
# Showing 3 best trials
# <keras_tuner.engine.objective.Objective object at 0x7fbe842c3e50>
# Trial summary
# Hyperparameters:
# num_layers: 3
# units: 30
# lr: 0.009263303924328512
# gamma: 0.11835983615085716
# Score: 0.9974660515785218
# Trial summary
# Hyperparameters:
# num_layers: 3
# units: 15
# lr: 0.0004000695122985253
# gamma: 0.06352943477687838
# Score: 0.9941176414489746
# Trial summary
# Hyperparameters:
# num_layers: 4
# units: 25
# lr: 0.0004926783901540332
# gamma: 1.6928927504819127
# Score: 0.992941164970398

#  tuner.search_space_summary()
# Search space summary
# Default search space size: 4
# num_layers (Int)
# {'default': None, 'conditions': [], 'min_value': 2, 'max_value': 4, 'step': 1, 'sampling': None}
# units (Int)
# {'default': None, 'conditions': [], 'min_value': 15, 'max_value': 30, 'step': 5, 'sampling': None}
# lr (Float)
# {'default': 0.0001, 'conditions': [], 'min_value': 0.0001, 'max_value': 0.03, 'step': None, 'sampling': 'log'}
# gamma (Float)
# {'default': 0.05, 'conditions': [], 'min_value': 0.05, 'max_value': 5.0, 'step': None, 'sampling': 'log'}


model = build_model(nNeurons=30, gamma=0.118359, lr=0.00926)
history = model.fit(
	X_train,
	y_train,
	epochs=45,
	validation_data=(X_test, y_test),
	callbacks=[callback],
	class_weight = class_weight)

#see visualizations by launching tensorboard in relevant logs dir!

#from the keras/tensorflow tutorial on unbalanced datasets:
def plot_metrics(history):
	metrics = ['loss', 'prc', 'precision', 'recall']
	for n, metric in enumerate(metrics):
		name = metric.replace("_"," ").capitalize()
		plt.subplot(2,2,n+1)
		plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
		plt.plot(history.epoch, history.history['val_'+metric],
			color=colors[0], linestyle="--", label='Val')
		plt.xlabel('Epoch')
		plt.ylabel(name)
		if metric == 'loss':
			plt.ylim([0, plt.ylim()[1]])
		elif metric == 'auc':
			plt.ylim([0.8,1])
		else:
			plt.ylim([0,1])
		plt.legend();

#also from keras/tensorflow tutorial pages on unbalanced datasets:
def plot_roc(name, labels, predictions, **kwargs):
	fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

	plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
	plt.xlabel('False positives [%]')
	plt.ylabel('True positives [%]')
	plt.xlim([-0.5,20])
	plt.ylim([80,100.5])
	plt.grid(True)
	ax = plt.gca()
	ax.set_aspect('equal')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plot_metrics(history)

plot_roc("Train", y_train, model.predict(X_train))
plt.show()
plot_roc("Test", y_test, model.predict(X_test), linestyle='--')
plt.legend(loc='lower right');
plt.show()


preds = (model.predict(X_test)>0.5).astype(int).ravel()
ConfusionMatrixDisplay.from_predictions(y_test, preds)
plt.show()

# for utility:
# model.save('./tuned_nn.tf')
# model = keras.models.load_model('./tuned_nn.tf')
