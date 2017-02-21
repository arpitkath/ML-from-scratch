from math import exp, sqrt
import numpy as np

def sigmoid(z):
	"""
	Argument : 
	1. z - The predited value y_predicted.

	Return : sigmoid or logistic of z
	"""
	return 1.0 / (1.0 + exp(-z))

def predict(X, y, W, b):
	"""
	Argument : 
	1. X - Input data.
	2. y - True values.
	3. W - Weights excluding bias term.
	4. b - bias term.

	Return : Output probability of y = 1.
	""" 
	y_predicted = b
	for i in xrange(len(W)):
		y_predicted += W[i] * X[i]

	return sigmoid(y_predicted)

def SGD(X, y, learning_rate, n_epochs):
	"""
	Argument :
	1. X - Input data.
	2. y - True output.
	3. learning_rate - The learning rate 'alpha' for Gradient Descent.
	4. n_epochs - Number of times the algorithms will be performed for input data.

	Return : 
	1. b - bias term
	2. W - Weights
	"""
	b = 0
	W = [0 for _ in xrange(len(X[0]))]

	for epoch in xrange(n_epochs):
		for i in xrange(len(X)):
			y_predicted = predict(X[i], y[i], W, b)
			error = y[i] - y_predicted
			b += learning_rate * error * y_predicted * (1.0 - y_predicted)
			for j in xrange(len(W)):
				W[j] += learning_rate * error * y_predicted * (1.0 - y_predicted) * X[i][j] # Not sure of this line

	return b, W

'''
AdaGrad is used for automatically learn the learning rate.
Every feature has its own learning rate, which depends on the history of gradient calculated in that direction.
Also, the learning rate for every dimension changes at every step, unlike in GGD, where it remains constant.
'''
def AdaGrad(X, y, n_epochs, fudge_factor=1e-6, step=1e-2):
	'''
	Arguments :
	fudge_factor - A threshold used to prevent divide by zero error.
	step - step size for change of weights.

	Return :
	W - Weights
	b - bias term.
	'''
	learning_rate = [0 for _ in xrange(len(X[0]))]
	b = 0
	W = list(learning_rate)

	for epoch in xrange(n_epochs):
		for i in xrange(len(X)):
			y_predicted = predict(X[i], y[i], W, b)
			error = y[i] - y_predicted
			b += step * error * y_predicted * (1.0 - y_predicted)
			gradient = [0] * len(W)
			for j in xrange(len(W)):
				gradient[j] = y_predicted * (1.0 - y_predicted) * X[i][j] * error
				learning_rate[j] += gradient[j] ** 2
				gradient[j] /= (fudge_factor + sqrt(learning_rate[j]))
				W[j] += step * gradient[j]

	return b, W

def RMSProp(X, y, n_epochs, fudge_factor=1e-6, step=1e-2, decay_rate=1e-1):

	learning_rate = [0 for _ in xrange(len(X[0]))]
	b = 0
	W = list(learning_rate)

	for epoch in xrange(n_epochs):
		for i in xrange(len(X)):
			y_predicted = predict(X[i], y[i], W, b)
			error = y[i] - y_predicted
			b += step * error * y_predicted * (1.0 - y_predicted)
			gradient = [0] * len(W)
			for j in xrange(len(W)):
				gradient[j] = y_predicted * (1.0 - y_predicted) * X[i][j] * error
				learning_rate[j] = (decay_rate * learning_rate[j]) + ( (1.0 - decay_rate) * gradient[j] ** 2 ) # Added decay_rate with AdaGrad.
				gradient[j] /= (fudge_factor + sqrt(learning_rate[j]))
				W[j] += step * gradient[j]

	return b, W

