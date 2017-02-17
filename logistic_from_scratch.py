from math import exp

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
		total_error = 0
		for i in xrange(len(X)):
			y_predicted = predict(X[i], y[i], W, b)
			error = y[i] - y_predicted
			total_error += error ** 2
			b += learning_rate * error * y_predicted * (1.0 - y_predicted)
			for j in xrange(len(W)):
				W[j] += learning_rate * error * y_predicted * (1.0 - y_predicted) * X[i][j]

		if epoch % 100 == 0:
			print "Epoch = %d, Error = %.2f, alpha = %.2f" % (epoch, total_error, learning_rate)

	return b, W


