import sklearn
import numpy as np
import matplotlib.pyplot as plt 
import sklearn.datasets

# Making dataset.
print "Dataset created."
np.random.seed(0)
X, y = sklearn.datasets.make_moons(2000, noise=0.3)

n_examples = len(X)
nn_input_dim = 2
nn_out_dim = 2
learning_rate = 0.01
reg_lambda = 0.01

def loss(model):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

	# Forward propagation.

	z1 = X.dot(W1) + b1
	# Usign tanh as activation function.
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	correct_logprobs = -np.log(probs[range(n_examples), y])
	_loss = np.sum(correct_logprobs)
	_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
	
	return 1. / n_examples*_loss


def predict(model, x):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

	z1 = x.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	return np.argmax(probs, axis=1)


def make_model(n_hidden, n_pass):

	np.random.seed(0)
	W1 = np.random.randn(nn_input_dim, n_hidden) / np.sqrt(nn_input_dim)
	b1 = np.zeros((1, n_hidden))
	W2 = np.random.randn(n_hidden, nn_out_dim) / np.sqrt(n_hidden)
	b2 = np.zeros((1, nn_out_dim))

	model = {}

	for i in xrange(n_pass):

		# Forward pass.
		z1 = X.dot(W1) + b1
		a1 = np.tanh(z1)
		z2 = a1.dot(W2) + b2
		exp_scores = np.exp(z2)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

		# Backprop.
		delta3 = probs
		delta3[range(n_examples), y] -= 1
		dW2 = (a1.T).dot(delta3)
		db2 = np.sum(delta3, axis=0, keepdims=True)
		delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
		dW1 = (X.T).dot(delta2)
		db1 = np.sum(delta2, axis=0)

		# Adding regularaization
		dW1 += reg_lambda * W1
		dW2 += reg_lambda * W2

		# Updating parameter using BGD

		W1 -= learning_rate * dW1
		W2 -= learning_rate * dW2
		b1 -= learning_rate * db1
		b2 -= learning_rate * db2

		model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

		if i % 1000 == 0:
			print  "Loss after iteration %i: %f" %(i, loss(model))

	return model

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    print(Z.shape, xx.shape)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer size %d' % nn_hdim)
    model = make_model(nn_hdim, 30000)
    plot_decision_boundary(lambda x: predict(model, x))
plt.show()
