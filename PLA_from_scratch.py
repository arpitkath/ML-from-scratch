import numpy as np

class Perceptron(object):

	def __init__(self, learning_rate, n_epochs):
		self.learning_rate = learning_rate
		self.n_epochs = n_epochs

	def fit(self, X, y):

		assert X.shape[0] == y.shape[0]
		self.W = np.zeros(X.shape[1])
		self.b = 0

		for _ in xrange(self.n_epochs):
			for i in xrange(X.shape[0]):
				self.b += self.learning_rate * (y[i] - self.predict(x))
				self.W += self.learning_rate * (y[i] - self.predict(x)) * X[i]

		return self

	def predict(self, x):
		res = np.dot(x, self.W) + self.b
		return np.where(res >= 0, 1, -1)

learning_rate = 0.1
n_epochs = 100
# X = 
# y = 
# X_ = 
perceptron = Perceptron(learning_rate, n_epochs)
perceptron.fit(X, y)
y_hat = perceptron.predict(X_)